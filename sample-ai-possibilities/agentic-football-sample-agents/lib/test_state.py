"""Tests for possession resolution in state.py.

Run directly:  python test_state.py

These cover the case where the same agentId appears on both teams, which is
how the game server numbers players (agentId_0..agentId_4 per team). Resolving
possession by index alone makes two players believe they hold the same ball.
"""

import copy
import json

from state import get_possession_info, holder_is_me, summarize_state
from test_helpers import GAME_STATE

HOME, AWAY = 0, 1
HOLDER_IDX = 3  # the fixture's ball is held by agentId_3 on the home team


def _ball(**overrides):
    b = copy.deepcopy(GAME_STATE["ball"])
    b.update(overrides)
    return b


def _players():
    return GAME_STATE["players"]


def test_fixture_has_colliding_agent_ids():
    """Guard the premise: agentId is not unique across teams."""
    ids = [p["agentId"] for p in _players()]
    assert len(ids) != len(set(ids)), "fixture no longer collides — revisit _resolve_holder"
    dupes = {i for i in ids if ids.count(i) > 1}
    print(f"=== PREMISE: agentIds shared across teams: {sorted(dupes)} ===")


def test_only_the_holder_reports_has_ball():
    """The opposing player with the same index must not claim the ball."""
    ball, players = _ball(), _players()
    assert holder_is_me(ball, players, HOME, HOLDER_IDX) is True
    assert holder_is_me(ball, players, AWAY, HOLDER_IDX) is False
    print("=== HOLDER: home p3 holds, away p3 does not ===")


def test_possession_side_is_per_team():
    """Same ball, same tick: each team sees the holder from its own side."""
    ball, players = _ball(), _players()
    assert get_possession_info(ball, players, HOME) == (HOLDER_IDX, "MY player 3", True)
    assert get_possession_info(ball, players, AWAY) == (HOLDER_IDX, "OPP player 3", False)
    print("=== SIDES: MY for home, OPP for away ===")


def test_rendered_prompt_is_not_self_contradictory():
    """The away forward used to read 'held by OPP player 3' and hasBall=True."""
    out = summarize_state(GAME_STATE, AWAY, HOLDER_IDX, "FWD1")
    assert "held by OPP player 3" in out
    assert "hasBall=False" in out, out
    print("=== PROMPT: away p3 told opponent holds it, and hasBall=False ===")


def test_server_isfree_is_honoured():
    """isFree is authoritative; it was previously ignored entirely."""
    assert get_possession_info(_ball(isFree=True), _players(), HOME) == (None, "free", False)
    print("=== ISFREE: respected ===")


def test_unresolvable_claim_is_unknown_not_free():
    """A claim we cannot parse must never render as a loose ball.

    Reporting 'free' here is what told a player carrying the ball to go and
    fetch it.
    """
    for bad in ("home_fwd1", "agentId_9", "arn:aws:bedrock:::agent/HomeFwd1"):
        got = get_possession_info(_ball(possessionAgentId=bad), _players(), HOME)
        assert got == (None, "unknown", False), (bad, got)
    print("=== UNRESOLVABLE: reported as unknown, not free ===")


def test_no_claim_is_free():
    ball = _ball()
    ball.pop("possessionAgentId", None)
    ball.pop("isFree", None)
    assert get_possession_info(ball, _players(), HOME) == (None, "free", False)
    print("=== NO CLAIM: genuinely loose ===")


def test_team_code_case_is_tolerated():
    """Uppercase teamCode must not erase the agent's own player.

    The server documents HOME/AWAY while the payload has been seen as
    home/away. Case sensitivity here drops the player from its own prompt and
    lists its teammates as opponents.
    """
    for transform in (str.upper, str.title):
        gs = copy.deepcopy(GAME_STATE)
        for p in gs["players"]:
            p["teamCode"] = transform(p["teamCode"])
        out = summarize_state(gs, AWAY, HOLDER_IDX, "FWD1")
        assert ">>> YOUR PLAYER" in out, f"own player missing with {transform.__name__} teamCode"
        assert "hasBall=False" in out
    print("=== TEAMCODE: upper/title case tolerated ===")


def _opponent_holds(idx):
    """Ball carried by the AWAY player agentId_<idx>, at that player's feet."""
    gs = copy.deepcopy(GAME_STATE)
    opp = next(p for p in gs["players"] if p["teamCode"] == "away" and p["agentId"] == f"agentId_{idx}")
    gs["ball"]["position"] = {**opp["position"], "z": 0}
    gs["ball"]["possessionAgentId"] = f"agentId_{idx}"
    return gs


def test_opponent_with_same_index_holding_ball_is_not_us():
    """The match-losing bug: away p<N> has the ball, home p<N> was told hasBall=True.

    players[] lists home first, so the old first-match rule resolved every
    ambiguous holder to us.
    """
    for idx in range(5):
        gs = _opponent_holds(idx)
        ball, players = gs["ball"], gs["players"]
        assert holder_is_me(ball, players, HOME, idx) is False, idx
        assert holder_is_me(ball, players, AWAY, idx) is True, idx
        assert get_possession_info(ball, players, HOME) == (idx, f"OPP player {idx}", False), idx
        out = summarize_state(gs, HOME, idx, "X")
        assert f"held by OPP player {idx}" in out and "hasBall=False" in out, out
    print("=== SAME-INDEX OPPONENT: never reported as our ball ===")


def test_possession_team_field_wins_over_distance():
    """If the server sends a team for the holder, trust it over geometry."""
    gs = _opponent_holds(3)
    gs["ball"]["possessionTeamCode"] = "HOME"  # ball sits at the away p3, but server says home
    assert get_possession_info(gs["ball"], gs["players"], HOME)[2] is True
    gs["ball"].pop("possessionTeamCode")
    gs["ball"]["possessionTeamId"] = 0
    assert get_possession_info(gs["ball"], gs["players"], HOME)[2] is True
    print("=== TEAM FIELD: authoritative when present ===")



# ---------------------------------------------------------------------------
# Server frame: real payload captured from a live match (1/10 scale, ball y =
# height, ball z = sideways). See normalize_game_state in state.py.
# ---------------------------------------------------------------------------
from state import normalize_game_state, to_server_commands, format_team_chat, is_server_frame

SERVER_SAMPLE = json.loads(r'''{"tick": 9634, "gameTime": 74.27955, "playMode": "OPEN_PLAY", "score": {"home": 4, "away": 0}, "ball": {"position": {"x": 1.7355746, "y": 0.14735724, "z": -0.4599825}, "velocity": {"x": 0, "y": 0, "z": 0}, "isFree": false, "possessionAgentId": "agentId_4"}, "players": [{"teamCode": "home", "agentId": "agentId_0", "position": {"x": -4.0876613, "y": 0.18641448}, "velocity": {"x": -0.99058926, "y": -0.13686845}, "stamina": 1}, {"teamCode": "home", "agentId": "agentId_1", "position": {"x": -2.9873185, "y": -0.32193464}, "velocity": {"x": 1.4993505, "y": -0.044134464}, "stamina": 0.80367386}, {"teamCode": "home", "agentId": "agentId_2", "position": {"x": 1.4831299, "y": -0.45352602}, "velocity": {"x": 0, "y": 0}, "stamina": 0.45859993}, {"teamCode": "home", "agentId": "agentId_3", "position": {"x": 0.5753527, "y": 1.1919796}, "velocity": {"x": 0, "y": 0}, "stamina": 0.70419157}, {"teamCode": "home", "agentId": "agentId_4", "position": {"x": -2.2168381, "y": -2.626974}, "velocity": {"x": 1.2933599, "y": 0.7597502}, "stamina": 0.80367386}, {"teamCode": "away", "agentId": "agentId_0", "position": {"x": 2.6112595, "y": -0.06823264}, "velocity": {"x": -0.99999976, "y": -0.00061800616}, "stamina": 1}, {"teamCode": "away", "agentId": "agentId_1", "position": {"x": 2.948415, "y": 0.7641377}, "velocity": {"x": -1.2093376, "y": -0.8874134}, "stamina": 0.8889699}, {"teamCode": "away", "agentId": "agentId_2", "position": {"x": 1.7014073, "y": -0.60529226}, "velocity": {"x": 0, "y": 0}, "stamina": 0.74872935}, {"teamCode": "away", "agentId": "agentId_3", "position": {"x": 5.288499, "y": 0.070565954}, "velocity": {"x": 0.9998878, "y": -0.014975703}, "stamina": 1}, {"teamCode": "away", "agentId": "agentId_4", "position": {"x": 1.3763336, "y": -0.42673802}, "velocity": {"x": 0, "y": 0}, "stamina": 0.9389044}], "teamChat": []}''')


def test_server_frame_is_scaled_to_documented_frame():
    gs = normalize_game_state(SERVER_SAMPLE)
    assert gs["_serverFrame"] is True
    home_gk = next(p for p in gs["players"] if p["teamCode"] == "home" and p["agentId"] == "agentId_0")
    assert abs(home_gk["position"]["x"] - (-40.88)) < 0.01, home_gk
    ball = gs["ball"]["position"]
    assert abs(ball["x"] - 17.36) < 0.01 and abs(ball["y"] - (-4.60)) < 0.01, ball   # sideways came from z
    assert abs(ball["z"] - 1.47) < 0.01, ball                                          # height moved to z
    assert SERVER_SAMPLE["ball"]["position"]["x"] == 1.7355746, "input must not be mutated"
    print("=== SERVER FRAME: x10 scale, ball z->y sideways ===")


def test_documented_frame_passes_through():
    gs = normalize_game_state(GAME_STATE)
    assert gs["_serverFrame"] is False and not is_server_frame(GAME_STATE)
    assert gs["ball"]["position"] == GAME_STATE["ball"]["position"]
    assert gs["players"][0]["position"] == GAME_STATE["players"][0]["position"]
    print("=== DOC FRAME: unchanged ===")


def test_missing_ball_position_reuses_last_known():
    normalize_game_state(SERVER_SAMPLE)                       # remembers the ball
    gap = copy.deepcopy(SERVER_SAMPLE)
    gap["ball"] = {"isFree": True, "possessionAgentId": None}  # as the server sends it
    gs = normalize_game_state(gap)
    assert gs["ball"]["position"]["x"] != 0 and gs["ball"].get("positionEstimated") is True
    print("=== MISSING BALL: last known position reused ===")


def test_commands_convert_back_to_server_frame():
    cmds = [{"commandType": "MOVE_TO", "parameters": {"target_x": -50.0, "target_y": 12.0, "sprint": True}},
            {"commandType": "FOLLOW_PLAYER", "parameters": {"target_player_id": 3, "target_team": "AWAY", "distance": 5.0}},
            {"commandType": "PRESS_BALL", "parameters": {"intensity": 0.8}}]
    out = to_server_commands(cmds, True)
    assert out[0]["parameters"] == {"target_x": -5.0, "target_y": 1.2, "sprint": True}
    assert out[1]["parameters"]["distance"] == 0.5 and out[1]["parameters"]["target_player_id"] == 3
    assert out[2]["parameters"]["intensity"] == 0.8
    assert to_server_commands(cmds, False) is cmds
    print("=== OUTGOING: distances /10, ids and intensities untouched ===")


def test_server_sample_summary_is_sane():
    gs = normalize_game_state(SERVER_SAMPLE)
    out = summarize_state(gs, HOME, 0, "GK")
    assert "pos=(-40.9,1.9)" in out, out          # GK near its own goal in the documented frame
    print("=== SUMMARY: GK reads as near its own goal ===")


def test_team_chat_reaches_the_prompt():
    gs = copy.deepcopy(GAME_STATE)
    gs["teamChat"] = [{"message": "press higher"}, "all-out attack now"]
    out = summarize_state(gs, HOME, 2, "MID")
    assert "COACH INSTRUCTIONS" in out and "press higher" in out and "all-out attack now" in out, out
    assert "COACH" not in summarize_state(GAME_STATE, HOME, 2, "MID")
    assert format_team_chat({"teamChat": []}) == []
    print("=== TEAM CHAT: coach lines rendered ===")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"{len(fns)} tests passed")
