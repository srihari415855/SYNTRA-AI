"""State Manager Module."""

from typing import Any, Dict, List, Optional
from app.schemas.agent_state import AgentState
from app.storage.repository import StateRepository, state_repository


class StateManager:
    """Manages active conversation state and transactional persistence checkpoints."""

    def __init__(self, repository: Optional[StateRepository] = None):
        self.repository = repository or state_repository

    async def initialize_state(self, session_id: str, user_id: str) -> AgentState:
        """Loads or creates session and initializes an AgentState."""
        await self.repository.get_or_create_session(session_id, user_id)
        history = await self.repository.get_session_context(session_id)
        return AgentState(
            session_id=session_id,
            current_context=history,
            final_status="IN_PROGRESS",
            iteration_count=0
        )

    async def record_user_prompt(self, session_id: str, prompt: str) -> None:
        """Records initial user prompt into state and persistence."""
        await self.repository.append_message(
            session_id=session_id,
            role="user",
            content=prompt
        )

    async def record_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result_content: str,
        state: AgentState
    ) -> None:
        """Appends tool execution result to message context and updates state."""
        await self.repository.append_message(
            session_id=session_id,
            role="tool",
            content=result_content,
            name=tool_name
        )
        state.update_timestamp()

    async def finalize_state(
        self,
        session_id: str,
        final_text: str,
        state: AgentState,
        status: str = "COMPLETE"
    ) -> None:
        """Records final assistant response and marks state status."""
        await self.repository.append_message(
            session_id=session_id,
            role="assistant",
            content=final_text
        )
        state.final_status = status
        await self.repository.update_session_state(state)


state_manager = StateManager()
