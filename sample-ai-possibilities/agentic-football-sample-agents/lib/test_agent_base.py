"""Handler tests (stub app, stub model). Run directly: python test_agent_base.py"""
import asyncio, json, copy
import state
from agent_base import create_invoke_handler
from fallback import build_fallback, GK_CONFIG, DEF_CONFIG
from test_helpers import GAME_STATE


class _App:
    def __init__(self):
        self.logs = []
        logs = self.logs
        class logger:
            info = staticmethod(lambda m: logs.append(m))
            warn = staticmethod(lambda m: logs.append(m))
            error = staticmethod(lambda m: logs.append(m))
        self.logger = logger

    def entrypoint(self, fn):
        self.fn = fn
        return fn


def _never_called(_):
    raise AssertionError("the model must not be called on an empty tick")


def _run(app, payload):
    async def go():
        return [x async for x in app.fn({"prompt": json.dumps(payload)}, None)]
    return json.loads(asyncio.run(go())[0])


EMPTY_TICK = {"tick": 0, "gameTime": 0, "playMode": "OPEN_PLAY", "score": {"home": 0, "away": 0},
              "ball": {"isFree": True, "possessionAgentId": None}, "players": [], "teamChat": []}


def test_empty_tick_skips_model_and_keeps_gk_home():
    state._last_frame.clear()   # first tick of a fresh container
    for team, goal in ((0, -5.2), (1, 5.2)):
        app = _App()
        create_invoke_handler(app, _never_called, 0, "GK", build_fallback(GK_CONFIG), fallback_cfg=GK_CONFIG)
        out = _run(app, {"gameState": EMPTY_TICK, "teamId": team, "myPlayers": [0]})
        assert out[0]["commandType"] == "MOVE_TO", out
        assert abs(out[0]["parameters"]["target_x"] - goal) < 0.01, out   # server units (x1/10)
        assert any(m.startswith("EMPTY GK") for m in app.logs)
    print("=== EMPTY TICK: GK goes to its goal, model not called ===")


def test_empty_tick_outfield_uses_fallback():
    app = _App()
    create_invoke_handler(app, _never_called, 1, "DEF", build_fallback(DEF_CONFIG), fallback_cfg=DEF_CONFIG)
    out = _run(app, {"gameState": EMPTY_TICK, "teamId": 0, "myPlayers": [1]})
    assert out and out[0]["commandType"] == "SET_STANCE", out
    print("=== EMPTY TICK: outfield player sends a safe stance ===")


def test_normal_tick_still_calls_model():
    calls = []
    def model(summary):
        calls.append(summary)
        return '[{"commandType":"MARK","playerId":1,"parameters":{"target_player_id":3,"tightness":"TIGHT"},"duration":3}]'
    app = _App()
    create_invoke_handler(app, model, 1, "DEF", build_fallback(DEF_CONFIG), fallback_cfg=DEF_CONFIG)
    out = _run(app, {"gameState": copy.deepcopy(GAME_STATE), "teamId": 0, "myPlayers": [1]})
    assert calls and out[0]["commandType"] == "MARK", out
    print("=== NORMAL TICK: model called as before ===")


def test_frame_remembered_across_empty_tick():
    server_tick = {"players": [{"agentId": "agentId_0", "teamCode": "home", "position": {"x": -5.5, "y": 0}}],
                   "ball": {"position": {"x": 0, "y": 0.15, "z": 0}}}
    assert state.normalize_game_state(server_tick)["_serverFrame"] is True
    assert state.normalize_game_state(EMPTY_TICK)["_serverFrame"] is True
    doc_tick = copy.deepcopy(GAME_STATE)
    assert state.normalize_game_state(doc_tick)["_serverFrame"] is False
    assert state.normalize_game_state(EMPTY_TICK)["_serverFrame"] is False
    print("=== FRAME: last measured frame reused on empty ticks ===")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"{len(fns)} tests passed")
