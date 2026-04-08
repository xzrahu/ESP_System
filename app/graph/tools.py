from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
from typing import Any

import vendor_bootstrap  # noqa: F401
from langchain_core.tools import tool

from graph.streaming import emit_global_event
from infrastructure.tools.local.ecommerce_product import (
    compare_products as compare_products_raw,
    get_product_detail as get_product_detail_raw,
    search_products as search_products_raw,
)
from infrastructure.tools.local.knowledge_base import query_knowledge as query_knowledge_raw
from infrastructure.tools.local.service_station import (
    query_nearest_repair_shops_by_coords as query_nearest_repair_shops_by_coords_raw,
    resolve_user_location_from_text as resolve_user_location_from_text_raw,
)
from infrastructure.tools.mcp.mcp_servers import baidu_map_mcp, search_mac_client


APP_DIR = Path(__file__).resolve().parents[1]
OFFLINE_NAVIGATION_SKILL_SCRIPT = (
    APP_DIR
    / "skills"
    / "offline-service-navigation-1.0.0"
    / "scripts"
    / "invoke_service_navigation.py"
)


def _extract_mcp_text(result: Any) -> str:
    fragments: list[str] = []
    for content in getattr(result, "content", []):
        text = getattr(content, "text", None)
        if text:
            fragments.append(text)
    return "\n".join(fragments).strip()


@lru_cache(maxsize=1)
def _load_offline_navigation_skill_module():
    spec = importlib.util.spec_from_file_location(
        "offline_service_navigation_skill",
        OFFLINE_NAVIGATION_SKILL_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load skill script: {OFFLINE_NAVIGATION_SKILL_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@tool("query_knowledge")
async def query_knowledge_tool(question: str) -> dict[str, Any]:
    """Query the knowledge base for technical support context."""
    await emit_global_event("tool", "query_knowledge")
    return await query_knowledge_raw(question)


@tool("bailian_web_search")
async def bailian_web_search_tool(query: str) -> str:
    """Search the web for real-time information."""
    await emit_global_event("tool", "bailian_web_search")
    result = await search_mac_client.call_tool("bailian_web_search", {"query": query})
    return _extract_mcp_text(result)


@tool("resolve_user_location_from_text")
async def resolve_user_location_from_text_tool(user_input: str) -> str:
    """Resolve the user's location from natural language."""
    await emit_global_event("tool", "resolve_user_location_from_text")
    return await resolve_user_location_from_text_raw(user_input)


@tool("query_nearest_repair_shops_by_coords")
def query_nearest_repair_shops_by_coords_tool(
    lat: float,
    lng: float,
    limit: int = 3,
) -> str:
    """Query nearby repair shops by coordinates."""
    return query_nearest_repair_shops_by_coords_raw(lat, lng, limit)


@tool("offline_service_navigation_skill")
async def offline_service_navigation_skill_tool(
    mode: str,
    query: str,
    brand: str = "",
    destination: str = "",
    limit: int = 3,
) -> dict[str, Any]:
    """Invoke the offline service and navigation skill.

    Use `service-station` to find nearby service shops.
    Use `poi-nav` for regular destination navigation.
    """
    await emit_global_event("tool", "offline_service_navigation_skill")
    skill_module = _load_offline_navigation_skill_module()

    if mode == "service-station":
        return await skill_module.run_service_station(
            query=query,
            brand=brand or None,
            limit=limit,
        )
    if mode == "poi-nav":
        if not destination.strip():
            return {
                "ok": False,
                "stage": "input_validation",
                "detail": "destination is required when mode=poi-nav",
            }
        return await skill_module.run_poi_nav(
            query=query,
            destination=destination,
        )
    return {
        "ok": False,
        "stage": "input_validation",
        "detail": f"unsupported mode: {mode}",
    }


@tool("geocode_address")
async def geocode_address_tool(address: str) -> str:
    """Resolve a destination address to Baidu map coordinates."""
    await emit_global_event("tool", "map_geocode")
    result = await baidu_map_mcp.call_tool("map_geocode", {"address": address})
    return _extract_mcp_text(result)


@tool("map_uri")
async def map_uri_tool(
    destination: str,
    origin: str = "",
) -> str:
    """Generate a Baidu map navigation link."""
    await emit_global_event("tool", "map_uri")
    arguments = {
        "service": "direction",
        "origin": origin,
        "destination": destination,
    }
    result = await baidu_map_mcp.call_tool("map_uri", arguments)
    return _extract_mcp_text(result)


@tool("search_products")
async def search_products_tool(keyword: str) -> dict[str, Any]:
    """Search products."""
    await emit_global_event("tool", "search_products")
    return await search_products_raw(keyword)


@tool("get_product_detail")
async def get_product_detail_tool(spu_id: int) -> dict[str, Any]:
    """Get product detail."""
    await emit_global_event("tool", "get_product_detail")
    return await get_product_detail_raw(spu_id)


@tool("compare_products")
async def compare_products_tool(query: str) -> dict[str, Any]:
    """Compare products."""
    await emit_global_event("tool", "compare_products")
    return await compare_products_raw(query)


TECHNICAL_TOOLS = [query_knowledge_tool, bailian_web_search_tool]
SERVICE_TOOLS = [
    offline_service_navigation_skill_tool,
    resolve_user_location_from_text_tool,
    query_nearest_repair_shops_by_coords_tool,
    geocode_address_tool,
    map_uri_tool,
]
PRODUCT_TOOLS = [search_products_tool, get_product_detail_tool, compare_products_tool]
