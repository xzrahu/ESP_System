from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import vendor_bootstrap  # noqa: F401
import httpx
import stun
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from config.settings import settings


def get_ip_via_stun() -> str | None:
    """Return the current public IP when it can be discovered."""
    try:
        _, external_ip, _ = stun.get_ip_info()
        return external_ip
    except Exception:
        return None


@dataclass(slots=True)
class _McpTransportConfig:
    transport: str
    url: str
    headers: dict[str, str] | None = None
    timeout: float = 60
    sse_read_timeout: float = 60 * 30


class MCPToolClient:
    """Small MCP client wrapper used by LangGraph tools."""

    def __init__(self, name: str, config: _McpTransportConfig):
        self.name = name
        self._config = config

    def _build_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=self._config.headers,
            timeout=httpx.Timeout(
                self._config.timeout,
                read=self._config.sse_read_timeout,
            ),
            follow_redirects=True,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        async with self._build_http_client() as http_client:
            if self._config.transport == "streamable_http":
                async with streamable_http_client(
                    self._config.url,
                    http_client=http_client,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
                return

            async with sse_client(
                self._config.url,
                headers=self._config.headers,
                timeout=self._config.timeout,
                sse_read_timeout=self._config.sse_read_timeout,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

    async def connect(self) -> None:
        async with self.session() as session:
            await session.list_tools()

    async def cleanup(self) -> None:
        return None

    async def list_tools(self) -> list[Any]:
        async with self.session() as session:
            result = await session.list_tools()
            return list(getattr(result, "tools", []))

    async def call_tool(
        self,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        *,
        name: str | None = None,
    ) -> Any:
        resolved_name = name or tool_name
        if not resolved_name:
            raise ValueError("tool_name is required")

        async with self.session() as session:
            return await session.call_tool(resolved_name, arguments=arguments or {})


search_mac_client = MCPToolClient(
    name="通用联网搜索",
    config=_McpTransportConfig(
        transport="streamable_http",
        url=settings.DASHSCOPE_BASE_URL or "",
        headers=(
            {"Authorization": f"Bearer {settings.AL_BAILIAN_API_KEY}"}
            if settings.AL_BAILIAN_API_KEY
            else None
        ),
    ),
)


baidu_map_mcp = MCPToolClient(
    name="百度地图",
    config=_McpTransportConfig(
        transport="sse",
        url=f"https://mcp.map.baidu.com/sse?ak={settings.BAIDUMAP_AK}",
    ),
)
