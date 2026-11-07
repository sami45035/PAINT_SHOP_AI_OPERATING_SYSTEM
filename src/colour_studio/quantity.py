"""
How much paint does the customer need, and which cans should they buy?
"""
import itertools
import math
from typing import Any, Dict, List, Optional

from src.config import DEFAULT_COVERAGE_SQFT_PER_LITRE_PER_COAT


def litres_needed(width_ft: float, height_ft: float, coats: int = 2, category: str = "Interior Emulsion",
                  coverage: Optional[float] = None, openings_sqft: float = 0.0) -> Dict[str, float]:
    cov = coverage or DEFAULT_COVERAGE_SQFT_PER_LITRE_PER_COAT.get(category, 100)
    area = max(0.0, width_ft * height_ft - openings_sqft)
    litres = area * max(1, coats) / cov
    return {"area_sqft": round(area, 1), "litres": round(litres, 2), "coverage": cov}


def best_packs(litres: float, packs: List[Dict[str, Any]], respect_stock: bool = True) -> Optional[Dict[str, Any]]:
    """
    Cheapest mix of cans that gives at least `litres`.
    packs: [{"size_litres", "price", "stock", "sku"}]. Returns
    {"items": [{"size_litres", "count", "price", "sku"}], "total_litres", "total_price"} or None.
    """
    if litres <= 0 or not packs:
        return None
    usable = [p for p in packs if p.get("price") and (not respect_stock or p.get("stock", 0) > 0)]
    if not usable:
        return None
    usable.sort(key=lambda p: p["size_litres"])
    ranges = []
    for p in usable:
        limit = math.ceil(litres / p["size_litres"]) + 1
        if respect_stock:
            limit = min(limit, int(p.get("stock", 0)))
        ranges.append(range(0, limit + 1))
    best = None
    for counts in itertools.product(*ranges):
        total_l = sum(c * p["size_litres"] for c, p in zip(counts, usable))
        if total_l + 1e-9 < litres:
            continue
        price = sum(c * p["price"] for c, p in zip(counts, usable))
        key = (price, total_l, sum(counts))
        if best is None or key < best[0]:
            best = (key, counts)
    if best is None:
        return None
    (price, total_l, _), counts = best
    items = [{"size_litres": p["size_litres"], "count": c, "price": p["price"], "sku": p.get("sku")}
             for c, p in zip(counts, usable) if c]
    items.sort(key=lambda i: -i["size_litres"])
    return {"items": items, "total_litres": total_l, "total_price": round(price, 2)}


def describe_packs(plan: Optional[Dict[str, Any]]) -> str:
    if not plan:
        return "Not enough stock to cover this wall."
    return " + ".join(f"{i['count']} × {i['size_litres']:g} L" for i in plan["items"])
