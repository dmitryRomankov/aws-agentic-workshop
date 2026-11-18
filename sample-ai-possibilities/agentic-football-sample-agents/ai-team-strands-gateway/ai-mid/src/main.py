"""
AI Soccer Midfielder Agent (Gateway) — Controls ONLY player 2 (Midfielder).
Uses Strands SDK + AgentCore Gateway MCP tools for tactical analysis.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from gateway_agent_base import create_gateway_agent
from gateway_invoke_handler import create_gateway_invoke_handler
from fallback import build_fallback, MID_CONFIG

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 2
POSITION_LABEL = "MID"

SYSTEM_PROMPT = f"""You are an AI soccer midfielder controlling ONLY player {MY_PLAYER_ID} in a 5v5 match.

You have tactical analysis TOOLS via MCP. Speed matters: every tool call adds about 1.5 seconds,
and a decision later than 5 seconds is lost.
## Tool budget — STRICT
- Call AT MOST ONE tool per tick. Never call a second tool.
- Call a tool ONLY if hasBall=True OR distBall < 15. Otherwise call NO tool and answer immediately.
- Which tool:
  - hasBall=True and distOppGoal < 25 → `evaluate_shot` (SHOOT if should_shoot, else PASS to the best open teammate)
  - hasBall=True otherwise → `calculate_pass_options`, then PASS to the top option
  - hasBall=False, distBall < 15, opponent has the ball → `get_defensive_assignment`
  - hasBall=False, distBall < 15, teammate has the ball or ball is free → `find_open_space`
- After the tool result, return the JSON command right away. No analysis text.
## Your Role — Midfielder
- You are the link between defense and attack — distribute the ball wisely
- Without a tool, decide from the game state: support the ball carrier, press a nearby opponent with the ball, or hold a central position
- Balance attack and defense — manage stamina carefully
## Available Commands
ONE-SHOT: MOVE_TO, PASS, SHOOT, SLIDE_TACKLE, GK_DISTRIBUTE
MAINTAINED: PRESS_BALL, MARK, INTERCEPT, FOLLOW_PLAYER
TACTICAL: SET_STANCE, CLEAR_OVERRIDE, RESET

## Field: x=-55 to +55, y=-35 to +35. Team 0 (HOME) defends -x.

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}.
Example: [{{"commandType":"PASS","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":3,"type":"THROUGH"}},"duration":0}}]
Return ONLY the JSON array, no text before or after."""

fallback_commands = build_fallback(MID_CONFIG)

agent, mcp_client = create_gateway_agent(
    SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
create_gateway_invoke_handler(
    app, agent, mcp_client, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=MID_CONFIG,
)

if __name__ == "__main__":
    app.run()
