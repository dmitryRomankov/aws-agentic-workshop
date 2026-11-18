"""Tests for tactics.py. Run directly: python test_tactics.py

States are in the documented frame (x ±55, y ±35), as the handler passes them
after normalize_game_state.
"""
from tactics import apply_tactics

HOME, AWAY = 0, 1


def _p(team, idx, x, y):
    return {"agentId": f"agentId_{idx}", "teamCode": team, "position": {"x": x, "y": y}, "stamina": 0.9}


def _state(home, away, holder=None, ball=None):
    """home/away: {idx: (x, y)}; holder: (team, idx) or None for a free ball."""
    players = [_p("home", i, *xy) for i, xy in home.items()] + [_p("away", i, *xy) for i, xy in away.items()]
    if holder:
        team, idx = holder
        bx, by = (home if team == "home" else away)[idx]
        ball = {"position": {"x": bx, "y": by, "z": 0}, "isFree": False, "possessionAgentId": f"agentId_{idx}"}
    else:
        ball = {"position": {"x": ball[0], "y": ball[1], "z": 0}, "isFree": True, "possessionAgentId": None}
    return {"playMode": "OPEN_PLAY", "ball": ball, "players": players}


def _c(ctype, pid, team, **params):
    return [{"commandType": ctype, "playerId": pid, "teamId": team, "parameters": params, "duration": 0}]


# The Summit Coyotes goal-2 situation: GK has the ball, DEF is marked.
GK_BALL = _state(
    home={0: (-52, 0), 1: (-40, 2), 2: (-20, 15), 3: (5, -20), 4: (0, 20)},
    away={0: (52, 0), 1: (-38, 3), 2: (-25, 12), 3: (-30, -5), 4: (2, 18)},
    holder=("home", 0))


def test_gk_kicks_to_most_advanced_open_teammate_not_marked_def():
    cmds, why = apply_tactics(_c("GK_DISTRIBUTE", 0, HOME, target_player_id=1, method="THROW"), GK_BALL, HOME, 0)
    # DEF(1) marked at 2.2, MID(2) marked at 5.8, FWD2(4) marked at 2.8 -> FWD1(3) at (5,-20) is open
    assert cmds[0]["commandType"] == "GK_DISTRIBUTE", cmds
    assert cmds[0]["parameters"] == {"target_player_id": 3, "method": "KICK"}, cmds
    assert cmds[0]["playerId"] == 0 and cmds[0]["teamId"] == HOME
    assert why, why
    print("=== GK: KICK to open FWD1, not the marked DEF ===")


def test_gk_rule_also_replaces_non_distribution_commands():
    cmds, _ = apply_tactics(_c("MOVE_TO", 0, HOME, target_x=-55, target_y=0, sprint=False), GK_BALL, HOME, 0)
    assert cmds[0]["commandType"] == "GK_DISTRIBUTE" and cmds[0]["parameters"]["target_player_id"] == 3
    print("=== GK: holding the ball always distributes ===")


KICKOFF = _state(  # our kickoff: FWD1 on the center spot, 55 from goal
    home={0: (-55, 0), 1: (-40, 0), 2: (-12, 19), 3: (0, -5), 4: (-12, -19)},
    away={0: (55, 0), 1: (40, 0), 2: (25, 0), 3: (12, 18), 4: (12, -18)},
    holder=("home", 3))


def test_long_shot_from_kickoff_becomes_short_pass():
    cmds, why = apply_tactics(_c("SHOOT", 3, HOME, aim_location="CENTER", power=0.95), KICKOFF, HOME, 3)
    assert cmds[0]["commandType"] == "PASS", cmds
    assert cmds[0]["parameters"]["target_player_id"] in (2, 4), cmds
    assert "SHOOT from 55" in why[0], why
    print("=== SHOOT: kickoff blast replaced by short pass ===")


def test_shot_in_range_is_kept():
    s = _state(home={0: (-55, 0), 1: (-30, 0), 2: (0, 0), 3: (35, 5), 4: (20, -10)},
               away={0: (52, 0), 1: (40, 5), 2: (30, 0), 3: (10, 0), 4: (5, 5)}, holder=("home", 3))
    orig = _c("SHOOT", 3, HOME, aim_location="TR", power=0.9)
    cmds, why = apply_tactics(orig, s, HOME, 3)
    assert cmds is orig and why == [], (cmds, why)
    print("=== SHOOT: 20.6 from goal kept ===")


def test_shot_from_34_is_kept():
    # match ef253aed: FWD1 at x=21 had its shot turned into a pass back to DEF
    s = _state(home={0: (-55, 0), 1: (-30, 0), 2: (0, 0), 3: (21, 3), 4: (10, -10)},
               away={0: (52, 0), 1: (35, 5), 2: (30, 0), 3: (10, 10), 4: (5, 5)}, holder=("home", 3))
    orig = _c("SHOOT", 3, HOME, aim_location="TR", power=0.9)
    cmds, why = apply_tactics(orig, s, HOME, 3)
    assert cmds is orig and why == [], (cmds, why)
    print("=== SHOOT: 34 from goal kept ===")


def test_forward_out_of_range_never_passes_back():
    # 43 from goal, only open teammates are behind -> dribble, not a back pass to DEF
    s = _state(home={0: (-55, 0), 1: (-30, 0), 2: (-5, 20), 3: (12, 0), 4: (5, -20)},
               away={0: (52, 0), 1: (30, 0), 2: (25, 5), 3: (40, 10), 4: (40, -10)}, holder=("home", 3))
    cmds, why = apply_tactics(_c("SHOOT", 3, HOME, aim_location="CENTER", power=0.9), s, HOME, 3)
    assert cmds[0]["commandType"] == "MOVE_TO" and cmds[0]["parameters"]["target_x"] > 12, cmds
    assert "SHOOT from 43" in why[0] and "dribble" in why[0], why
    print("=== SHOOT: forward at 43 dribbles instead of passing back ===")


def test_forward_out_of_range_passes_to_open_teammate_ahead():
    s = _state(home={0: (-55, 0), 1: (-30, 0), 2: (-5, 20), 3: (12, 0), 4: (25, -20)},
               away={0: (52, 0), 1: (30, 0), 2: (25, 5), 3: (40, 10), 4: (40, 10)}, holder=("home", 3))
    cmds, _ = apply_tactics(_c("SHOOT", 3, HOME, aim_location="CENTER", power=0.9), s, HOME, 3)
    assert cmds[0]["commandType"] == "PASS" and cmds[0]["parameters"]["target_player_id"] == 4, cmds
    print("=== SHOOT: forward at 43 passes to FWD2 ahead ===")


def test_defender_never_shoots():
    s = _state(home={0: (-55, 0), 1: (-30, 0), 2: (-10, 5), 3: (10, -5), 4: (15, 15)},
               away={0: (55, 0), 1: (30, 0), 2: (20, 10), 3: (-20, 0), 4: (-25, -10)}, holder=("home", 1))
    cmds, why = apply_tactics(_c("SHOOT", 1, HOME, aim_location="CENTER", power=1.0), s, HOME, 1)
    assert cmds[0]["commandType"] == "PASS" and "GK/DEF may not shoot" in why[0], (cmds, why)
    print("=== SHOOT: DEF shot replaced ===")


BUILD_OUT = _state(  # Coyotes goal-1 situation: MID with the ball near our corner
    home={0: (-55, 0), 1: (-40, 25), 2: (-33, 29), 3: (-11, 0), 4: (5, -15)},
    away={0: (55, 0), 1: (-14, 3), 2: (-30, 22), 3: (-38, 20), 4: (20, 0)},
    holder=("home", 2))


def test_build_out_back_pass_becomes_aerial_forward():
    cmds, why = apply_tactics(_c("PASS", 2, HOME, target_player_id=1, type="GROUND"), BUILD_OUT, HOME, 2)
    # DEF(1) is behind; FWD1(3) is marked by away 1 at 4.2; FWD2(4) at (5,-15) is open and ahead
    assert cmds[0]["parameters"] == {"target_player_id": 4, "type": "AERIAL"}, cmds
    assert "own-third" in why[0]
    print("=== BUILD-OUT: back pass -> AERIAL to open FWD2 ===")


def test_build_out_forward_pass_to_open_mate_is_kept():
    orig = _c("PASS", 2, HOME, target_player_id=4, type="GROUND")
    cmds, why = apply_tactics(orig, BUILD_OUT, HOME, 2)
    assert cmds is orig and why == []
    print("=== BUILD-OUT: good forward pass kept ===")


PRESS = _state(  # opponent 3 carries the ball at our box; 3 of ours want to press
    home={0: (-54, 0), 1: (-38, 5), 2: (-30, 0), 3: (-25, -8), 4: (-20, 12)},
    away={0: (54, 0), 1: (-10, 0), 2: (-28, 15), 3: (-40, 3), 4: (-33, -12)},
    holder=("away", 3))


def test_only_nearest_presses_others_mark_distinct_attackers():
    out = {}
    for pid in (1, 2, 3, 4):
        cmds, _ = apply_tactics(_c("PRESS_BALL", pid, HOME, intensity=0.8), PRESS, HOME, pid)
        out[pid] = cmds[0]
    pressers = [pid for pid, c in out.items() if c["commandType"] == "PRESS_BALL"]
    assert pressers == [1], out                        # DEF is 2.8 from the carrier
    marks = [c["parameters"]["target_player_id"] for c in out.values() if c["commandType"] == "MARK"]
    assert len(marks) == len(set(marks)) == 3, out     # three different attackers, no doubling up
    assert 3 not in marks and 0 not in marks           # never the carrier or their GK
    print(f"=== PRESS: P1 presses, marks={sorted(marks)} ===")


def test_press_on_free_ball_is_left_alone():
    s = _state(home=PRESS_HOME, away=PRESS_AWAY, ball=(-30, 0))
    orig = _c("PRESS_BALL", 4, HOME, intensity=0.7)
    cmds, why = apply_tactics(orig, s, HOME, 4)
    assert cmds is orig and why == []
    print("=== PRESS: free ball unchanged ===")


PRESS_HOME = {0: (-54, 0), 1: (-38, 5), 2: (-30, 0), 3: (-25, -8), 4: (-20, 12)}
PRESS_AWAY = {0: (54, 0), 1: (-10, 0), 2: (-28, 15), 3: (-40, 3), 4: (-33, -12)}


def test_away_team_is_mirrored():
    # Same GK situation mirrored for the AWAY side (we defend +x).
    mir = lambda d: {i: (-x, y) for i, (x, y) in d.items()}
    s = _state(home=mir({0: (52, 0), 1: (-38, 3), 2: (-25, 12), 3: (-30, -5), 4: (2, 18)}),
               away=mir({0: (-52, 0), 1: (-40, 2), 2: (-20, 15), 3: (5, -20), 4: (0, 20)}),
               holder=("away", 0))
    cmds, _ = apply_tactics(_c("GK_DISTRIBUTE", 0, AWAY, target_player_id=1, method="THROW"), s, AWAY, 0)
    assert cmds[0]["parameters"] == {"target_player_id": 3, "method": "KICK"}, cmds
    print("=== AWAY: mirrored GK choice identical ===")


def test_other_commands_untouched():
    orig = _c("MOVE_TO", 3, HOME, target_x=20, target_y=0, sprint=True)
    cmds, why = apply_tactics(orig, KICKOFF, HOME, 4)
    assert cmds is orig and why == []
    print("=== OTHER: MOVE_TO untouched ===")



# ---------------------------------------------------------------------------
# GK leash (the SheQL match: GK sat on the center spot, we conceded 4)
# ---------------------------------------------------------------------------
MIDFIELD = _state(
    home={0: (-50, 0), 1: (-30, 0), 2: (-5, 5), 3: (10, -5), 4: (12, 12)},
    away={0: (52, 0), 1: (30, 0), 2: (0, 0), 3: (-10, 10), 4: (-12, -12)},
    holder=("away", 2))


def test_gk_move_to_center_is_clamped_to_its_area():
    cmds, why = apply_tactics(_c("MOVE_TO", 0, HOME, target_x=0.0, target_y=0.0, sprint=False), MIDFIELD, HOME, 0)
    assert cmds[0]["commandType"] == "MOVE_TO" and cmds[0]["parameters"]["target_x"] == -43.0, cmds
    assert "clamped" in why[0]
    print("=== LEASH: GK MOVE_TO (0,0) -> x=-43 ===")


def test_gk_leash_mirrored_for_away():
    mir = lambda d: {i: (-x, y) for i, (x, y) in d.items()}
    s = _state(home=mir({0: (52, 0), 1: (30, 0), 2: (0, 0), 3: (-10, 10), 4: (-12, -12)}),
               away=mir({0: (-50, 0), 1: (-30, 0), 2: (-5, 5), 3: (10, -5), 4: (12, 12)}), holder=("home", 2))
    cmds, _ = apply_tactics(_c("MOVE_TO", 0, AWAY, target_x=0.0, target_y=-1.7, sprint=False), s, AWAY, 0)
    assert cmds[0]["parameters"]["target_x"] == 43.0, cmds
    print("=== LEASH: AWAY GK MOVE_TO (0,-1.7) -> x=+43 ===")


def test_gk_intercept_far_from_goal_goes_back_to_line():
    cmds, why = apply_tactics(_c("INTERCEPT", 0, HOME, aggressive=True), MIDFIELD, HOME, 0)
    assert cmds[0]["commandType"] == "MOVE_TO" and cmds[0]["parameters"]["target_x"] == -52.0, cmds
    print("=== LEASH: GK INTERCEPT at midfield -> back to the line ===")


def test_gk_intercept_near_goal_is_kept():
    s = _state(home={0: (-50, 0), 1: (-30, 0), 2: (-5, 5), 3: (10, -5), 4: (12, 12)},
               away={0: (52, 0), 1: (30, 0), 2: (0, 0), 3: (-10, 10), 4: (-12, -12)}, ball=(-42, 4))
    orig = _c("INTERCEPT", 0, HOME, aggressive=True)
    cmds, why = apply_tactics(orig, s, HOME, 0)
    assert cmds is orig and why == []
    print("=== LEASH: GK INTERCEPT 13 from goal kept ===")


def test_gk_move_inside_area_is_kept():
    orig = _c("MOVE_TO", 0, HOME, target_x=-50.0, target_y=3.0, sprint=False)
    cmds, why = apply_tactics(orig, MIDFIELD, HOME, 0)
    assert cmds is orig and why == []
    print("=== LEASH: GK MOVE_TO (-50,3) kept ===")

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"{len(fns)} tests passed")
