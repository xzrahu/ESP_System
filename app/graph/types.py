from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


AgentKind = Literal["technical", "service", "product"]


class TaskItem(TypedDict):
    agent_type: AgentKind
    query: str
    reason: str


class GraphState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    remaining_steps: int
    llm_input_messages: list[Any]
    user_id: str
    session_id: str
    user_query: str
    conversation_summary: str
    memory_context: str
    tasks: list[TaskItem]
    current_task_index: int
    specialist_outputs: list[dict[str, str]]
    final_answer: str
