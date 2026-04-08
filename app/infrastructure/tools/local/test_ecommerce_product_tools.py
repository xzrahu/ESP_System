import argparse
import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List


APP_ROOT = Path(__file__).resolve().parents[3]


def _install_agents_stub() -> None:
    """
    Allow this regression script to run even when the `agents` package
    is not installed in the current Python environment.
    """
    if "agents" in sys.modules:
        return

    mock_agents = types.ModuleType("agents")
    mock_agents.function_tool = lambda f: f
    sys.modules["agents"] = mock_agents


def _load_tools():
    _install_agents_stub()
    sys.path.insert(0, str(APP_ROOT))

    from infrastructure.tools.local.ecommerce_product import (
        _compare_products,
        _get_product_detail,
        _search_products,
    )

    return _search_products, _get_product_detail, _compare_products


def _extract_spu_list(search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(search_result, dict):
        return []

    data = search_result.get("data")
    if isinstance(data, dict):
        spu_list = data.get("spuList")
        if isinstance(spu_list, list):
            return [item for item in spu_list if isinstance(item, dict)]

    return []


def _extract_cps_list(search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(search_result, dict):
        return []

    data = search_result.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    return []


def _summarize_search_result(search_result: Dict[str, Any]) -> Dict[str, Any]:
    spu_list = _extract_spu_list(search_result)
    cps_list = _extract_cps_list(search_result)
    first = cps_list[0] if cps_list else (spu_list[0] if spu_list else {})
    return {
        "errno": search_result.get("errno"),
        "errmsg": search_result.get("errmsg"),
        "searchSource": search_result.get("searchSource"),
        "linkType": search_result.get("linkType"),
        "result_count": len(cps_list) if cps_list else len(spu_list),
        "first_product": {
            "productName": first.get("productName") or first.get("goodsName") or first.get("title"),
            "spuId": first.get("spuId"),
            "price": first.get("spuPrice") or first.get("price"),
            "shopName": first.get("shopName") or first.get("mallName") or first.get("source"),
            "link": first.get("spuUrl") or first.get("goodsUrl") or first.get("materialUrl"),
        } if first else None,
    }


def _summarize_detail_result(detail_result: Dict[str, Any]) -> Dict[str, Any]:
    raw_data = detail_result.get("data", {}) if isinstance(detail_result, dict) else {}
    data = raw_data if isinstance(raw_data, dict) else {}
    sku_list = data.get("skuList", []) if isinstance(data, dict) else []
    attributes = data.get("attributes", []) if isinstance(data, dict) else []
    return {
        "errno": detail_result.get("errno"),
        "errmsg": detail_result.get("errmsg"),
        "productName": data.get("productName"),
        "spuId": data.get("spuId"),
        "shopName": data.get("shopName"),
        "spuPrice": data.get("spuPrice"),
        "spuUrl": data.get("spuUrl"),
        "sku_count": len(sku_list) if isinstance(sku_list, list) else 0,
        "attribute_count": len(attributes) if isinstance(attributes, list) else 0,
    }


def _summarize_compare_result(compare_result: Dict[str, Any]) -> Dict[str, Any]:
    raw_data = compare_result.get("data", {}) if isinstance(compare_result, dict) else {}
    data = raw_data if isinstance(raw_data, dict) else {}
    spu_list = data.get("spuList", []) if isinstance(data, dict) else []
    full_compare_list = data.get("fullCompareList", []) if isinstance(data, dict) else []

    products = []
    if isinstance(spu_list, list):
        for item in spu_list[:2]:
            if isinstance(item, dict):
                products.append(
                    {
                        "title": item.get("title"),
                        "price": item.get("price"),
                        "priceText": item.get("priceText"),
                    }
                )

    tabs = []
    if isinstance(full_compare_list, list):
        for item in full_compare_list[:5]:
            if isinstance(item, dict):
                tabs.append(item.get("tabName"))

    return {
        "errno": compare_result.get("errno"),
        "errmsg": compare_result.get("errmsg"),
        "products": products,
        "tab_names": tabs,
        "tab_count": len(full_compare_list) if isinstance(full_compare_list, list) else 0,
    }


async def run_regression(search_keyword: str, compare_query: str) -> int:
    search_products, get_product_detail, compare_products = _load_tools()

    print("=" * 80)
    print("Ecommerce Product Tools Regression Test")
    print("=" * 80)
    print(f"Search keyword: {search_keyword}")
    print(f"Compare query: {compare_query}")

    failures: List[str] = []

    print("\n[1/3] Testing search_products")
    search_result = await search_products(search_keyword)
    search_summary = _summarize_search_result(search_result)
    print(json.dumps(search_summary, ensure_ascii=False, indent=2))

    spu_list = _extract_spu_list(search_result)
    cps_list = _extract_cps_list(search_result)
    if search_result.get("errno") != 0:
        failures.append("search_products returned non-zero errno")
    if not spu_list and not cps_list:
        failures.append("search_products returned no cps or spu items")

    detail_result: Dict[str, Any] = {"skipped": True}
    first_spu_id = None
    if spu_list:
        first_spu_id = spu_list[0].get("spuId")

    print("\n[2/3] Testing get_product_detail")
    if first_spu_id:
        detail_result = await get_product_detail(int(first_spu_id))
        detail_summary = _summarize_detail_result(detail_result)
        print(json.dumps(detail_summary, ensure_ascii=False, indent=2))

        if detail_result.get("errno") != 0:
            failures.append("get_product_detail returned non-zero errno")
        if not detail_summary.get("productName"):
            failures.append("get_product_detail returned no productName")
        if not detail_summary.get("spuUrl"):
            failures.append("get_product_detail returned no spuUrl")
    else:
        failures.append("could not obtain spuId from search_products")
        print(json.dumps(detail_result, ensure_ascii=False, indent=2))

    print("\n[3/3] Testing compare_products")
    compare_result = await compare_products(compare_query)
    compare_summary = _summarize_compare_result(compare_result)
    print(json.dumps(compare_summary, ensure_ascii=False, indent=2))

    if compare_result.get("errno") != 0:
        failures.append("compare_products returned non-zero errno")
    if not compare_summary.get("products"):
        failures.append("compare_products returned no products")
    if compare_summary.get("tab_count", 0) <= 0:
        failures.append("compare_products returned no compare tabs")

    print("\n" + "=" * 80)
    if failures:
        print("Regression result: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Regression result: PASSED")
    if first_spu_id:
        print(f"Verified detail lookup with spuId={first_spu_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression test for ecommerce product tools")
    parser.add_argument(
        "--search-keyword",
        default="小米 14 手机",
        help="Keyword used for search_products",
    )
    parser.add_argument(
        "--compare-query",
        default="iPhone 16和iPhone 15对比",
        help="Query used for compare_products",
    )
    args = parser.parse_args()

    return asyncio.run(run_regression(args.search_keyword, args.compare_query))


if __name__ == "__main__":
    raise SystemExit(main())
