from __future__ import annotations

from contextvars import ContextVar, Token
import inspect
from collections.abc import Awaitable, Callable

from langchain_core.runnables import RunnableConfig


EventCallback = Callable[[str, str], Awaitable[None] | None]
_event_callback_var: ContextVar[EventCallback | None] = ContextVar(
    "graph_event_callback",
    default=None,
)


async def emit_graph_event(config: RunnableConfig | None, kind: str, text: str) -> None:
    if not config:
        return

    configurable = config.get("configurable", {})
    callback: EventCallback | None = configurable.get("emit_event")
    if callback is None:
        return

    result = callback(kind, text)
    if inspect.isawaitable(result):
        await result


def chunk_text(text: str, chunk_size: int = 96) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def set_event_callback(callback: EventCallback | None) -> Token:
    return _event_callback_var.set(callback)


def reset_event_callback(token: Token) -> None:
    _event_callback_var.reset(token)


async def emit_global_event(kind: str, text: str) -> None:
    callback = _event_callback_var.get()
    if callback is None:
        return

    result = callback(kind, text)
    if inspect.isawaitable(result):
        await result
