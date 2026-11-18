"""Base agent factory for AI soccer position agents."""

import json
import time
from typing import Callable
from strands import Agent
from strands.models import BedrockModel

from parsing import parse_commands
from state import summarize_state, holder_is_me, dist, _is_my_team, normalize_game_state, to_server_commands, has_player_positions, get_goal_positions
from fallback import FallbackConfig, build_last_resort
from tactics import apply_tactics


def create_agent(system_prompt: str, model_id: str = "us.amazon.nova-micro-v1:0") -> Agent:
    """Create a Strands Agent with the given system prompt."""
    model = BedrockModel(model_id=model_id)
    return Agent(model=model, system_prompt=system_prompt)


def parse_prompt(payload) -> dict:
    """Parse the invocation payload's prompt, accepting either str or dict.

    The main path has always accepted both, but the error handler used to
    re-parse with a bare json.loads(), which raises TypeError on a dict. That
    exception was caught by the outer bare except, so the rule-based fallback
    was skipped entirely and control fell through to the last resort.
    """
    prompt = payload.get("prompt", "{}")
    if isinstance(prompt, dict):
        return prompt
    try:
        parsed = json.loads(prompt)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def team_id_from(prompt_data: dict, log, label: str) -> int:
    """teamId from the payload; HOME only as a loudly logged last resort.

    teamId is authoritative input, not a guessable default. Silently
    defaulting to 0 makes an away agent emit commands tagged HOME, which the
    server attributes to the opposing team — and leaves nothing in the logs
    to explain why the away side played as though it were the home side.
    """
    team_id = prompt_data.get("teamId")
    if team_id is None:
        log.error(
            f"{label}: payload contained no teamId; falling back to HOME (0). "
            "Commands this tick may be attributed to the wrong team."
        )
        return 0
    return team_id


def _tick_diag(game_state: dict, team_id: int, pid: int) -> dict:
    """What this player saw: enough to tell why the server ignored a PASS/SHOOT.

    Logged once per tick as a single 'DIAG {json}' line. Candidates are every
    player whose agentId matches the server's possession claim (one per team
    when the id is ambiguous), with their distance to the ball.
    """
    ball = game_state.get("ball") or {}
    bpos = ball.get("position") or {}
    players = game_state.get("players") or []
    claim = ball.get("possessionAgentId")
    cands = [
        {"team": p.get("teamCode"), "me": _is_my_team(p, team_id),
         "x": round((p.get("position") or {}).get("x", 0), 1),
         "y": round((p.get("position") or {}).get("y", 0), 1),
         "d": round(dist(p.get("position") or {}, bpos), 2)}
        for p in players if claim is not None and p.get("agentId") == claim
    ]
    return {
        "tick": game_state.get("tick"), "t": game_state.get("gameTime"),
        "mode": game_state.get("playMode"),
        "ball": [round(bpos.get("x", 0), 1), round(bpos.get("y", 0), 1)],
        "vel": [round((ball.get("velocity") or {}).get("x", 0), 1), round((ball.get("velocity") or {}).get("y", 0), 1)],
        "free": ball.get("isFree"), "claim": claim, "cands": cands,
        "hasBall": holder_is_me(ball, players, team_id, pid),
    }


BALL_ACTIONS = {"PASS", "SHOOT", "GK_DISTRIBUTE"}
RAW_SAMPLE_EVERY = 30  # log the full game state every N ticks per container
_tick_counter: list = [0]

def _empty_tick_commands(fallback_fn, game_state: dict, team_id: int, pid: int) -> list:
    """Safe command when the tick has no player positions."""
    if pid == 0:
        my_goal_x, _ = get_goal_positions(team_id)
        return [{"commandType": "MOVE_TO", "playerId": 0, "teamId": team_id, "duration": 0,
                 # 3 units in front of the goal line, same spot the GK leash uses
                 "parameters": {"target_x": my_goal_x - 3.0 if my_goal_x > 0 else my_goal_x + 3.0,
                                "target_y": 0.0, "sprint": False}}]
    return fallback_fn(game_state, team_id, pid)


_payload_shape_logged: list = []  # module-level once-flag (list so the closure can mutate it)


def create_invoke_handler(
    app,
    agent: Agent,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
):
    """Create and register the @app.entrypoint invoke handler.

    Three layers of error handling, from best to worst:
      1. LLM response → parse into commands
      2. fallback_fn(game_state, team_id, my_player_id) → rule-based commands
      3. last-resort command from fallback_cfg → single safe command
    """
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        # Parsed once, outside the try, so the error handler below can reuse
        # it instead of re-parsing and throwing a second exception.
        prompt_data = parse_prompt(payload)
        team_id = team_id_from(prompt_data, log, position_label)
        try:
            raw_game_state = prompt_data.get("gameState", {})
            # Convert the server's 1/10-scale, y-is-height frame to the documented
            # frame every prompt and fallback rule is written for.
            game_state = normalize_game_state(raw_game_state)
            server_frame = game_state.get("_serverFrame", False)
            if not _payload_shape_logged:
                # Once per container: which fields the server sends, so
                # possession resolution can use a team field if one exists.
                _payload_shape_logged.append(True)
                ball = game_state.get("ball") or {}
                possession = {k: ball[k] for k in ball if k.startswith("possession") or k == "isFree"}
                log.info(f"{position_label} payload shape: top={sorted(prompt_data)} "
                         f"ball={possession} ballKeys={sorted(ball)}")

            # Honor myPlayers from payload if present, otherwise use configured player ID
            my_players = prompt_data.get("myPlayers", [my_player_id])
            effective_pid = my_players[0] if my_players else my_player_id

            if not has_player_positions(game_state):
                # The first tick of every match has no players. Asking the model
                # here produced "MOVE_TO (0, 0)" for the GK in 2 of 17 matches,
                # and memory then repeated it all match (0-4 vs SheQL). Do not
                # call the model and do not write to memory on such a tick.
                commands = _empty_tick_commands(fallback_fn, game_state, team_id, effective_pid)
                log.info(f"EMPTY {position_label}: no player positions; sent "
                         f"{[c.get('commandType') for c in commands]} without calling the model")
                yield json.dumps(to_server_commands(commands, server_frame))
                return
            state_summary = summarize_state(
                game_state, team_id, effective_pid, position_label
            )
            log.info(f"{position_label} agent invoked for team {team_id}, controlling player {effective_pid}")

            t_start = time.time()
            response = agent(state_summary)
            response_text = str(response)

            def on_recovered(raw: str) -> None:
                # The model wrote Python-flavoured JSON (usually `True`/`False`/`None`).
                # We recovered it rather than dropping the command and falling back —
                # logged so you can see how often your model does this.
                log.warn(f"{position_label} recovered malformed JSON from the model: {raw[:200]}")

            commands = parse_commands(response_text, team_id, effective_pid, on_recovered)

            src = "llm" if commands else "fallback"
            # Guard: a ball action without the ball is ignored by the server and
            # wastes the tick (the GK sent ~20 GK_DISTRIBUTE per match this way).
            if commands and commands[0].get("commandType") in BALL_ACTIONS:
                ball = game_state.get("ball") or {}
                if not holder_is_me(ball, game_state.get("players") or [], team_id, effective_pid):
                    log.info(f"GUARD {position_label}: dropped {commands[0].get('commandType')} without the ball")
                    commands = fallback_fn(game_state, team_id, effective_pid)
                    src = "guard"
            if commands:
                try:  # code-enforced tactics (see tactics.py); never cost a tick
                    commands, reasons = apply_tactics(commands, game_state, team_id, effective_pid)
                    if reasons:
                        log.info(f"TACTIC {position_label}: " + " | ".join(reasons))
                        src = f"{src}+tactic"
                except Exception as tactic_error:
                    log.warn(f"TACTIC failed: {tactic_error}")
            _tick_counter[0] += 1
            if _tick_counter[0] % RAW_SAMPLE_EVERY == 1:
                try:  # Full raw game state, to confirm field scale and axes.
                    log.info("RAW " + json.dumps({"pos": position_label, "teamId": team_id,
                                                  "gameState": raw_game_state}, separators=(",", ":"), default=str))
                except Exception as raw_error:
                    log.warn(f"RAW failed: {raw_error}")
            try:
                diag = _tick_diag(game_state, team_id, effective_pid)
                diag.update(frame="server" if server_frame else "doc", pos=position_label, ms=int((time.time() - t_start) * 1000),
                            src=src,
                            cmd=[{"type": c.get("commandType"), "p": c.get("parameters")} for c in commands])
                log.info("DIAG " + json.dumps(diag, separators=(",", ":"), default=str))
            except Exception as diag_error:  # diagnostics must never cost a tick
                log.warn(f"DIAG failed: {diag_error}")
            if commands:
                log.info(f"LLM returned {len(commands)} commands: "
                         f"{[c.get('commandType') for c in commands]}")
                yield json.dumps(to_server_commands(commands, server_frame))
            else:
                log.warn(f"LLM parse failed, using fallback. Response: {response_text[:200]}")
                commands = fallback_fn(game_state, team_id, effective_pid)
                log.info(f"Fallback returned {len(commands)} commands")
                yield json.dumps(to_server_commands(commands, server_frame))

        except Exception as e:
            log.error(f"{position_label} agent error: {e}")
            try:
                # Reuse the payload parsed at entry. Re-parsing here is what
                # broke this path: json.loads() on an already-decoded dict
                # raises TypeError, the bare except below swallowed it, and
                # the rule-based fallback never ran.
                my_players = prompt_data.get("myPlayers", [my_player_id])
                effective_pid = my_players[0] if my_players else my_player_id
                gs_norm = normalize_game_state(prompt_data.get("gameState", {}))
                commands = fallback_fn(gs_norm, team_id, effective_pid)
                yield json.dumps(to_server_commands(commands, gs_norm.get("_serverFrame", False)))
            except Exception as fallback_error:
                log.error(f"{position_label} fallback also failed: {fallback_error}")
                cmd = dict(last_resort)
                # team_id came from the payload at entry, so the last resort
                # is no longer hardcoded to HOME.
                cmd["teamId"] = team_id
                yield json.dumps([cmd])

    return invoke
