"""Memory-aware agent factory for AI soccer position agents.

Extends the shared lib's agent_base with AgentCoreMemorySessionManager
so each agent persists conversation history across game ticks.
"""

import os
import uuid
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig


# Messages sent to the model each tick: the last 5 ticks (user + assistant).
# Without an explicit cap the prompt grows with every tick of the match.
HISTORY_WINDOW = 10


def create_memory_agent(
    system_prompt: str,
    player_id: int,
    position_label: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
) -> Agent:
    """Create a Strands Agent backed by AgentCore Memory (STM).

    Required env vars:
      MEMORY_ID  — AgentCore Memory resource ID
      TEAM_ID    — used as actor_id and session_id prefix
    """
    memory_id = os.environ.get("MEMORY_ID")
    team_id = os.environ.get("TEAM_ID", "default-team")

    if not memory_id:
        raise RuntimeError("MEMORY_ID environment variable is required")

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id,
            # One memory session per runtime instance. AgentCore runs each
            # runtime session (one per match) in its own instance, so this
            # scopes memory to the match. A fixed id replayed every earlier
            # match on start-up (~2,400 events per player), pushing ticks to 6-26s.
            session_id=f"match-{team_id}-{position_label}-{uuid.uuid4().hex[:12]}",
            actor_id=f"{team_id}-{position_label}",
        ),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

    model = BedrockModel(model_id=model_id)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        session_manager=session_manager,
        conversation_manager=SlidingWindowConversationManager(window_size=HISTORY_WINDOW),
    )
