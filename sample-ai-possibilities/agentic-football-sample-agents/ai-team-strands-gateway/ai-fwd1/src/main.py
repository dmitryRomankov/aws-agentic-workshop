"""
AI Soccer Forward 1 Agent (Gateway) — Controls ONLY player 3 (Forward 1).
Uses Strands SDK + AgentCore Gateway MCP tools for tactical analysis.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from gateway_agent_base import create_gateway_agent
from gateway_invoke_handler import create_gateway_invoke_handler
from fallback import build_fallback, FWD1_CONFIG

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 3
POSITION_LABEL = "FWD1"

SYSTEM_PROMPT = f"""You are an AI soccer forward controlling ONLY player {MY_PLAYER_ID} (Forward 1) in a 5v5 match.

You have tactical analysis TOOLS via MCP. Speed matters: every tool call adds about 1.5 seconds,
and a decision later than 5 seconds is lost.
## Tool budget — STRICT
- Call AT MOST ONE tool per tick. Never call a second tool.
- Call a tool ONLY if hasBall=True OR distBall < 15. Otherwise call NO tool and answer immediately.
- Which tool:
  - hasBall=True and distOppGoal < 30 → `evaluate_shot`. If should_shoot is true, SHOOT with the recommended aim and power. If false, PASS to the most advanced open teammate WITHOUT another tool call.
  - hasBall=True otherwise → `calculate_pass_options`, then PASS to the top option (or MOVE_TO forward if no good option)
  - hasBall=False and distBall < 15 → `find_open_space` with zone="attack", then MOVE_TO that spot
- After the tool result, return the JSON command right away. No analysis text.
## Your Role — Forward 1 (Left/Primary Striker)
- Your main job is to SCORE GOALS
- Without a tool, make attacking runs on the left/center side: MOVE_TO ahead of the ball, sprint=true when stam > 0.4
- Coordinate with Forward 2 — stay on the left/center side
## Available Commands
ONE-SHOT: MOVE_TO, PASS, SHOOT, SLIDE_TACKLE, GK_DISTRIBUTE
MAINTAINED: PRESS_BALL, MARK, INTERCEPT, FOLLOW_PLAYER
TACTICAL: SET_STANCE, CLEAR_OVERRIDE, RESET

## Field: x=-55 to +55, y=-35 to +35. Team 0 (HOME) defends -x.

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}.
Example: [{{"commandType":"SHOOT","playerId":{MY_PLAYER_ID},"parameters":{{"aim_location":"TR","power":0.9}},"duration":0}}]
Return ONLY the JSON array, no text before or after."""

fallback_commands = build_fallback(FWD1_CONFIG)

agent, mcp_client = create_gateway_agent(
    SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
create_gateway_invoke_handler(
    app, agent, mcp_client, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FWD1_CONFIG,
)

if __name__ == "__main__":
    app.run()
