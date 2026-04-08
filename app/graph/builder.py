from __future__ import annotations

import asyncio
from typing import Any

import vendor_bootstrap  # noqa: F401
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph_supervisor import create_supervisor

from graph.agents import product_agent, service_agent, technical_agent
from graph.memory import memory_service
from graph.models import build_main_chat_model, build_sub_chat_model
from graph.streaming import emit_graph_event
from graph.types import GraphState
from infrastructure.ai.prompt_loader import load_prompt


summary_llm = build_sub_chat_model(streaming=False)

SUPERVISOR_PROMPT = (
    load_prompt("orchestrator_v1")
    + "\n\nYou are the primary supervisor for three specialists:"
    + "\n- technical_specialist: troubleshooting, diagnostics, and how-to issues."
    + "\n- service_specialist: service stations, offline service flows, navigation, and location-aware support."
    + "\n- product_specialist: product lookup, specs, pricing, and comparisons."
    + "\n\nDelegate when a specialist is needed. Include a concise task description with each transfer."
    + "\nIf one specialist is enough, wait for the result and answer the user directly."
    + "\nIf multiple domains are involved, coordinate them one at a time and then produce a final user-facing answer yourself."
)

SUMMARY_PROMPT = (
    "Summarize the older conversation history briefly. Keep durable user goals, device details, "
    "decisions already made, and unresolved issues. Do not invent facts."
)

_graph_lock = asyncio.Lock()
_compiled_graph: Any | None = None


def _get_visible_messages(state: GraphState) -> list[Any]:
    return list(state.get("messages", []))


def _get_latest_user_query(state: GraphState) -> str:
    user_query = str(state.get("user_query", "") or "").strip()
    if user_query:
        return user_query

    for message in reversed(_get_visible_messages(state)):
        if isinstance(message, HumanMessage) and message.content:
            return str(message.content).strip()
    return ""


async def supervisor_pre_model_hook(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    messages = _get_visible_messages(state)
    updates: dict[str, Any] = {}

    summary = str(state.get("conversation_summary", "") or "").strip()
    old_messages = memory_service.build_summary_candidate(messages)
    if old_messages:
        response = await summary_llm.ainvoke(
            [
                SystemMessage(content=SUMMARY_PROMPT),
                HumanMessage(
                    content="\n\n".join(
                        f"{message.type}: {message.content}" for message in old_messages
                    )
                ),
            ],
            config=config,
        )
        summary = str(response.content or "").strip()
        if summary:
            updates["conversation_summary"] = summary

    memory_context = str(state.get("memory_context", "") or "").strip()
    user_id = str(state.get("user_id", "") or "").strip()
    user_query = _get_latest_user_query(state)
    if user_id and user_query:
        memory_context = await memory_service.recall_memories(user_id, user_query)
        updates["memory_context"] = memory_context
        if memory_context:
            await emit_graph_event(config, "process", "Loaded long-term memory context")

    context_blocks: list[str] = []
    if summary:
        context_blocks.append(f"Short-term conversation summary:\n{summary}")
    if memory_context:
        context_blocks.append(f"Long-term memory:\n{memory_context}")

    if context_blocks:
        updates["llm_input_messages"] = [
            SystemMessage(
                content="Additional conversation context for routing and answer synthesis:\n\n"
                + "\n\n".join(context_blocks)
            ),
            *messages,
        ]
    else:
        updates["llm_input_messages"] = messages

    return updates


def build_graph():
    if memory_service.checkpointer is None or memory_service.store is None:
        raise RuntimeError("MemoryService must be initialized before building the graph.")

    workflow = create_supervisor(
        agents=[technical_agent, service_agent, product_agent],
        model=build_main_chat_model(streaming=False),
        prompt=SUPERVISOR_PROMPT,
        state_schema=GraphState,
        output_mode="last_message",
        supervisor_name="main_supervisor",
        handoff_tool_prefix="delegate_to",
        pre_model_hook=supervisor_pre_model_hook,
    )

    return workflow.compile(
        checkpointer=memory_service.checkpointer,
        store=memory_service.store,
    )


async def get_chat_graph():
    global _compiled_graph

    if _compiled_graph is not None:
        return _compiled_graph

    await memory_service.initialize()
    async with _graph_lock:
        if _compiled_graph is None:
            _compiled_graph = build_graph()
    return _compiled_graph


def reset_chat_graph() -> None:
    global _compiled_graph
    _compiled_graph = None


class LazyChatGraph:
    async def ainvoke(self, *args, **kwargs):
        graph = await get_chat_graph()
        return await graph.ainvoke(*args, **kwargs)


chat_graph = LazyChatGraph()
