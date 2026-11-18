"""Code-enforced tactical rules, applied after the model has chosen a command.

The models repeatedly ignored prompt rules in real matches, and each of these
cost goals (match vs Summit Coyotes, 0-2):

  * GK distribution  — the GK threw all 6 distributions to the nearest
    teammate, a defender who was being marked; 5 of 6 were lost inside our
    third and the second goal came straight from that.
  * Shooting         — 5 of 7 shots were FWD1 shooting from the kickoff spot,
    55 units out, giving the ball away after every restart.
  * Building out     — DEF/MID played short sideways/backward passes in our own
    third; one was intercepted and scored from.
  * Pressing         — 3-4 players pressed the same carrier near our box and
    left the other attackers free.

All coordinates here are in the documented frame (x ±55, y ±35), i.e. after
state.normalize_game_state. Every agent sees the same game state each tick,
so the marking assignment below comes out the same in all five agents
without them talking to each other.
"""
import math

from state import _player_idx, _is_my_team, _resolve_holder, holder_is_me, dist

OPEN_DIST = 10.0        # a teammate is "open" if no opponent is this close
SHOOT_RANGE = 30.0      # max distance to the opponent goal for SHOOT
OWN_THIRD = -18.0       # toward-opponent x below this = our defensive third
GOAL_X = 55.0
DRIBBLE_STEP = 15.0
TIGHT_MARK_RANGE = 35.0  # mark TIGHT when the attacker is this close to our goal


def _pos(p: dict) -> dict:
    return p.get("position") or {}


def _adv(p: dict, team_id: int) -> float:
    """x measured toward the opponent goal (+55 = their goal, -55 = ours)."""
    x = _pos(p).get("x", 0)
    return x if team_id == 0 else -x


def _from_adv(adv: float, team_id: int) -> float:
    return adv if team_id == 0 else -adv


def _cmd(orig: dict, ctype: str, params: dict, duration: int = 0) -> dict:
    c = {k: orig[k] for k in ("playerId", "teamId") if k in orig}
    c.update(commandType=ctype, parameters=params, duration=duration)
    return c


class _View:
    def __init__(self, game_state: dict, team_id: int, pid: int):
        self.team_id = team_id
        self.pid = pid
        self.ball = game_state.get("ball") or {}
        self.ball_pos = self.ball.get("position") or {"x": 0, "y": 0}
        self.players = [p for p in game_state.get("players") or [] if "x" in _pos(p)]
        ours = [p for p in self.players if _is_my_team(p, team_id)]
        self.me = next((p for p in ours if _player_idx(p) == pid), None)
        self.mates = [p for p in ours if _player_idx(p) != pid]
        self.opps = [p for p in self.players if not _is_my_team(p, team_id)]
        self.has_ball = holder_is_me(self.ball, game_state.get("players") or [], team_id, pid)
        holder = _resolve_holder(self.ball, game_state.get("players") or [])
        self.opp_has_ball = (self.ball.get("isFree") is not True and holder is not None
                             and not _is_my_team(holder, team_id))
        self.carrier = holder if self.opp_has_ball else None

    def adv(self, p):
        return _adv(p, self.team_id)

    def nearest_opp(self, p) -> float:
        return min((dist(_pos(p), _pos(o)) for o in self.opps), default=99.0)

    def is_open(self, p) -> bool:
        return self.nearest_opp(p) >= OPEN_DIST

    def dist_to_opp_goal(self, p) -> float:
        return math.hypot(GOAL_X - self.adv(p), _pos(p).get("y", 0))

    def dist_to_my_goal(self, p) -> float:
        return math.hypot(GOAL_X + self.adv(p), _pos(p).get("y", 0))

    def field_mates(self):
        """Teammates who can receive a pass in open play (never the GK)."""
        return [p for p in self.mates if _player_idx(p) != 0]

    def best_open_ahead(self, margin: float = 2.0):
        me_adv = self.adv(self.me)
        c = [p for p in self.field_mates() if self.is_open(p) and self.adv(p) > me_adv + margin]
        return max(c, key=self.adv, default=None)

    def nearest_open(self):
        """Nearest open teammate, preferring players outside our defensive third
        (a pass back into our own third restarts the risky build-out)."""
        c = [p for p in self.field_mates() if self.is_open(p)]
        upfield = [p for p in c if self.adv(p) >= OWN_THIRD]
        return min(upfield or c, key=lambda p: dist(_pos(p), _pos(self.me)), default=None)


# ---------------------------------------------------------------------------
# Rules. Each returns (new_command, reason) or (None, None) to leave it alone.
# ---------------------------------------------------------------------------
def _rule_gk_distribution(v: _View, c: dict):
    if v.pid != 0 or not v.has_ball:
        return None, None
    cands = v.field_mates()
    if not cands:
        return None, None
    open_ = [p for p in cands if v.is_open(p)]
    target = max(open_, key=v.adv) if open_ else max(cands, key=v.nearest_opp)
    tid = _player_idx(target)
    p = c.get("parameters") or {}
    if (c.get("commandType") == "GK_DISTRIBUTE" and p.get("target_player_id") == tid
            and p.get("method") == "KICK"):
        return None, None
    new = _cmd(c, "GK_DISTRIBUTE", {"target_player_id": tid, "method": "KICK"})
    return new, (f"GK {c.get('commandType')}->{p.get('target_player_id', '-')} replaced by KICK to "
                 f"{tid} ({'open' if open_ else 'least marked'}, nearest opp {v.nearest_opp(target):.0f})")


def _pass_or_dribble(v: _View, c: dict):
    """Best forward option for a player on the ball."""
    own_third = v.adv(v.me) < OWN_THIRD
    target = v.best_open_ahead() or v.nearest_open()
    if target is not None:
        ptype = "AERIAL" if own_third else "GROUND"
        return _cmd(c, "PASS", {"target_player_id": _player_idx(target), "type": ptype}), \
            f"PASS {ptype} to {_player_idx(target)}"
    me = _pos(v.me)
    tx = _from_adv(min(v.adv(v.me) + DRIBBLE_STEP, GOAL_X - 5), v.team_id)
    return _cmd(c, "MOVE_TO", {"target_x": round(tx, 1), "target_y": round(me.get("y", 0) * 0.5, 1),
                               "sprint": True}), "dribble forward (no open teammate)"


def _rule_shooting(v: _View, c: dict):
    if c.get("commandType") != "SHOOT" or not v.has_ball:
        return None, None
    d = v.dist_to_opp_goal(v.me)
    if v.pid not in (0, 1) and d <= SHOOT_RANGE:
        return None, None
    new, what = _pass_or_dribble(v, c)
    why = "GK/DEF may not shoot" if v.pid in (0, 1) else f"SHOOT from {d:.0f} > {SHOOT_RANGE:.0f}"
    return new, f"{why} -> {what}"


def _rule_build_out(v: _View, c: dict):
    if v.pid not in (1, 2) or not v.has_ball or c.get("commandType") != "PASS":
        return None, None
    if v.adv(v.me) >= OWN_THIRD:
        return None, None
    p = c.get("parameters") or {}
    tgt = next((m for m in v.field_mates() if _player_idx(m) == p.get("target_player_id")), None)
    if tgt is not None and v.is_open(tgt) and v.adv(tgt) > v.adv(v.me) + 5:
        return None, None   # forward pass to an open teammate: fine
    better = v.best_open_ahead(margin=5)
    if better is not None:
        new = _cmd(c, "PASS", {"target_player_id": _player_idx(better), "type": "AERIAL"})
        return new, (f"own-third PASS to {p.get('target_player_id')} (back/sideways/marked) -> "
                     f"AERIAL to {_player_idx(better)}")
    if p.get("type") != "AERIAL":
        new = _cmd(c, "PASS", dict(p, type="AERIAL"))
        return new, "own-third PASS, nobody open ahead -> same target, AERIAL"
    return None, None


def _rule_pressing(v: _View, c: dict):
    if c.get("commandType") != "PRESS_BALL" or not v.opp_has_ball or v.pid == 0:
        return None, None
    outfield = [p for p in v.mates + [v.me] if _player_idx(p) != 0]
    bp = v.ball_pos
    presser = min(outfield, key=lambda p: (dist(_pos(p), bp), _player_idx(p)))
    if _player_idx(presser) == v.pid:
        return None, None
    # Deterministic marking: most dangerous free attacker first, nearest of us.
    carrier_idx = _player_idx(v.carrier) if v.carrier else None
    attackers = sorted([o for o in v.opps if _player_idx(o) not in (0, carrier_idx)],
                       key=v.dist_to_my_goal)
    free = sorted([p for p in outfield if p is not presser], key=_player_idx)
    for o in attackers:
        if not free:
            break
        mine = min(free, key=lambda p: (dist(_pos(p), _pos(o)), _player_idx(p)))
        free.remove(mine)
        if _player_idx(mine) == v.pid:
            tight = "TIGHT" if v.dist_to_my_goal(o) < TIGHT_MARK_RANGE else "LOOSE"
            new = _cmd(c, "MARK", {"target_player_id": _player_idx(o), "tightness": tight}, duration=3)
            return new, f"not nearest to ball (P{_player_idx(presser)} presses) -> MARK {_player_idx(o)} {tight}"
    # Nobody left to mark: cover the line from the ball to our goal.
    gx = _from_adv(-GOAL_X, v.team_id)
    tx = bp.get("x", 0) + (gx - bp.get("x", 0)) * 0.4
    ty = bp.get("y", 0) * 0.6
    new = _cmd(c, "MOVE_TO", {"target_x": round(tx, 1), "target_y": round(ty, 1), "sprint": False})
    return new, f"not nearest to ball (P{_player_idx(presser)} presses), nobody to mark -> cover"


GK_LEASH = 12.0          # how far off its goal line the GK may move (toward the field)
GK_LEASH_Y = 10.0        # and sideways
GK_ENGAGE_RANGE = 20.0   # the GK leaves its line for the ball only this close to our goal
GK_ENGAGE = {"INTERCEPT", "PRESS_BALL", "MARK", "FOLLOW_PLAYER", "SLIDE_TACKLE"}


def _gk_line_move(v: _View, c: dict, reason: str):
    by = v.ball_pos.get("y", 0)
    tx = _from_adv(-GOAL_X + 3, v.team_id)
    new = _cmd(c, "MOVE_TO", {"target_x": round(tx, 1), "target_y": round(max(-6.0, min(6.0, by * 0.3)), 1),
                              "sprint": False})
    return new, reason


def _rule_gk_leash(v: _View, c: dict):
    """Keep the GK at its goal whatever the model says (it once sat on the
    center spot for a whole match and we conceded 4)."""
    if v.pid != 0 or v.has_ball:
        return None, None
    ctype = c.get("commandType")
    if ctype in GK_ENGAGE:
        d = v.dist_to_my_goal({"position": v.ball_pos})
        if d <= GK_ENGAGE_RANGE:
            return None, None
        return _gk_line_move(v, c, f"GK {ctype} with ball {d:.0f} from our goal -> back to the line")
    if ctype != "MOVE_TO":
        return None, None
    p = dict(c.get("parameters") or {})
    tx, ty = p.get("target_x"), p.get("target_y")
    if not isinstance(tx, (int, float)) or not isinstance(ty, (int, float)):
        return _gk_line_move(v, c, "GK MOVE_TO without a target -> back to the line")
    adv = tx if v.team_id == 0 else -tx
    cadv = max(-GOAL_X, min(-GOAL_X + GK_LEASH, adv))
    cy = max(-GK_LEASH_Y, min(GK_LEASH_Y, ty))
    if abs(cadv - adv) < 0.05 and abs(cy - ty) < 0.05:
        return None, None
    p.update(target_x=round(_from_adv(cadv, v.team_id), 1), target_y=round(cy, 1))
    return _cmd(c, "MOVE_TO", p), f"GK MOVE_TO ({tx:.0f},{ty:.0f}) outside its area -> clamped to ({p['target_x']},{p['target_y']})"


RULES = (_rule_gk_distribution, _rule_gk_leash, _rule_shooting, _rule_build_out, _rule_pressing)


def apply_tactics(commands: list, game_state: dict, team_id: int, pid: int):
    """Return (commands, reasons). Only the first command is checked."""
    if not commands:
        return commands, []
    v = _View(game_state, team_id, pid)
    if v.me is None:
        return commands, []
    first, reasons = commands[0], []
    for rule in RULES:
        new, why = rule(v, first)
        if new is not None:
            first, reasons = new, reasons + [why]
    return ([first] + list(commands[1:])) if reasons else commands, reasons
