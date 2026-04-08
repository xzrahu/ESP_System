from __future__ import annotations

from typing import Any

import vendor_bootstrap  # noqa: F401
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from graph.models import build_sub_chat_model
from graph.tools import PRODUCT_TOOLS, SERVICE_TOOLS, TECHNICAL_TOOLS
from graph.types import GraphState
from infrastructure.ai.prompt_loader import load_prompt


def _build_contextual_prompt(base_prompt: str):
    def prompt(state: dict[str, Any]) -> list[Any]:
        blocks = [base_prompt]

        summary = str(state.get("conversation_summary", "") or "").strip()
        if summary:
            blocks.append(f"Short-term conversation summary:\n{summary}")

        memory_context = str(state.get("memory_context", "") or "").strip()
        if memory_context:
            blocks.append(f"Long-term memory:\n{memory_context}")

        return [SystemMessage(content="\n\n".join(blocks)), *list(state.get("messages", []))]

    return prompt


SERVICE_PROMPT = (
    load_prompt("comprehensive_service_agent")
    + "\n\n新增规则:\n"
    + "1. 处理服务站查询或普通导航时，优先调用 `offline_service_navigation_skill`。\n"
    + "2. 只有当 skill 返回失败或信息不足时，才回退到底层地图或位置工具。\n"
    + "3. 服务站需求使用 mode=`service-station`，普通导航使用 mode=`poi-nav`。"
)


technical_agent = create_react_agent(
    model=build_sub_chat_model(),
    tools=TECHNICAL_TOOLS,
    prompt=_build_contextual_prompt(load_prompt("technical_agent")),
    name="technical_specialist",
    state_schema=GraphState,
)


service_agent = create_react_agent(
    model=build_sub_chat_model(),
    tools=SERVICE_TOOLS,
    prompt=_build_contextual_prompt(SERVICE_PROMPT),
    name="service_specialist",
    state_schema=GraphState,
)


product_agent = create_react_agent(
    model=build_sub_chat_model(),
    tools=PRODUCT_TOOLS,
    prompt=_build_contextual_prompt(load_prompt("product_query_agent")),
    name="product_specialist",
    state_schema=GraphState,
)
