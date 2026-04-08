import os
from pathlib import Path
from typing import Any, Dict

import httpx

from infrastructure.logging.logger import logger


BAIDU_ECOMMERCE_BASE_URL = "https://mcp-youxuan.baidu.com/skill"
APP_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def _read_token_from_env_file() -> str | None:
    if not APP_ENV_PATH.exists():
        return None

    for line in APP_ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "BAIDU_EC_SEARCH_TOKEN":
            return value.strip().strip('"').strip("'")
    return None


def _get_baidu_ecommerce_token() -> str:
    token = os.getenv("BAIDU_EC_SEARCH_TOKEN") or _read_token_from_env_file()
    if not token:
        raise ValueError("未配置 BAIDU_EC_SEARCH_TOKEN，无法使用商品查询能力。")
    return token


async def _request_baidu_ecommerce(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    token = _get_baidu_ecommerce_token()
    query = {**params, "key": token}
    url = f"{BAIDU_ECOMMERCE_BASE_URL}/{endpoint}"

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
            response = await client.get(
                url,
                params=query,
                headers={"User-Agent": "ITS-Product-Query-Agent/1.0"},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(f"百度电商接口 HTTP 错误: endpoint={endpoint}, error={exc}")
        return {
            "errno": exc.response.status_code,
            "errmsg": f"百度电商接口请求失败: {exc.response.status_code}",
        }
    except httpx.HTTPError as exc:
        logger.error(f"百度电商接口请求异常: endpoint={endpoint}, error={exc}")
        return {"errno": -1, "errmsg": f"百度电商接口不可用: {exc}"}
    except ValueError as exc:
        logger.error(str(exc))
        return {"errno": -1, "errmsg": str(exc)}


def _has_cps_items(result: Dict[str, Any]) -> bool:
    data = result.get("data")
    return isinstance(data, list) and len(data) > 0


def _has_spu_items(result: Dict[str, Any]) -> bool:
    data = result.get("data")
    return isinstance(data, dict) and isinstance(data.get("spuList"), list) and len(data.get("spuList")) > 0


async def _search_products(keyword: str) -> Dict[str, Any]:
    """
    优先使用 CPS 商品搜索，返回全网购买链接。
    如果 CPS 没有结果，再回退到百度优选 spu_list，保留详情查询能力。
    """
    if not keyword or not keyword.strip():
        return {"errno": -1, "errmsg": "商品关键词不能为空"}

    query = keyword.strip()

    cps_result = await _request_baidu_ecommerce("goods_search", {"query": query})
    if cps_result.get("errno") == 0 and _has_cps_items(cps_result):
        cps_result["searchSource"] = "cps_goods_search"
        cps_result["linkType"] = "purchase_link"
        return cps_result

    spu_result = await _request_baidu_ecommerce("spu_list", {"query": query})
    if spu_result.get("errno") == 0 and _has_spu_items(spu_result):
        spu_result["searchSource"] = "spu_list_fallback"
        spu_result["linkType"] = "product_detail_link"
        return spu_result

    if cps_result.get("errno") == 0:
        cps_result["searchSource"] = "cps_goods_search"
        cps_result["linkType"] = "purchase_link"
        return cps_result

    return spu_result


async def _get_product_detail(spu_id: int) -> Dict[str, Any]:
    if not spu_id:
        return {"errno": -1, "errmsg": "商品ID不能为空"}
    return await _request_baidu_ecommerce("spu_detail", {"spuId": spu_id})


async def _compare_products(query: str) -> Dict[str, Any]:
    if not query or not query.strip():
        return {"errno": -1, "errmsg": "商品对比请求不能为空"}
    return await _request_baidu_ecommerce("param_compare", {"query": query.strip()})


search_products = _search_products
get_product_detail = _get_product_detail
compare_products = _compare_products
