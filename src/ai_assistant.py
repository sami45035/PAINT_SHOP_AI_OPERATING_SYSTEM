"""
Ask Aura — conversational shop assistant (Layer 7)

Uses Gemini when an API key is set; otherwise answers common questions offline
from the same live data the rest of the app uses.
"""
import os
import re
from typing import Any, Dict, List, Optional

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.database import get_inventory_status, get_sales_summary, get_season_status
from src.predictor import compute_sales_forecast
from src.smart_alerts import flatten_alerts, generate_smart_alerts
from src.formatting import rupees as _rupees
from src.colour_assistant import search_shades_by_query

try:
    from google import genai
    from google.genai import types as genai_types

    GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    GENAI_AVAILABLE = False

SYSTEM_INSTRUCTION = """
You are Aura, the shop assistant for an independent paint shop in India.
You are given live shop data below: stock, sales, reorder plan, alerts and season.
Rules:
- Answer only from the data given. If the data does not contain the answer, say so plainly.
- Use plain words a shop owner uses ("cans", "bills", "visitors"), not jargon.
- Be brief: a short answer, then a small list or table if it helps.
- Money in rupees with Indian grouping (₹2,20,868).
"""

STOPWORDS = {
    "do", "we", "have", "in", "stock", "check", "for", "is", "the", "are", "there", "any", "a", "an",
    "of", "how", "many", "much", "left", "cans", "can", "available", "our", "what", "about", "show", "me",
}


# --- Tools: each returns plain markdown built from live data ---------------------

def tool_check_stock(query: str) -> str:
    df = get_inventory_status()
    tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in STOPWORDS]
    if not tokens:
        return "Which shade or brand should I check? For example: *Do we have Royale Blue?*"
    hay = (df["brand"] + " " + df["shade_name"] + " " + df["category"] + " " + df["sku"]).str.lower()
    mask = hay.apply(lambda h: all(t in h for t in tokens))
    if not mask.any():  # fall back to any-token match
        mask = hay.apply(lambda h: any(t in h for t in tokens))
    matches = df[mask].sort_values(["shade_name", "pack_size_litres"])
    if matches.empty:
        return f"I couldn't find anything matching “{query}” in stock."
    lines = [f"Found {len(matches)} item{'s' if len(matches) != 1 else ''}:", "",
             "| Product | Pack | On shelf | Price | Shelf |", "|---|---|---|---|---|"]
    for r in matches.head(10).itertuples(index=False):
        flag = " (low)" if r.current_stock <= r.min_threshold else ""
        lines.append(f"| {r.brand} {r.shade_name} | {r.pack_size_litres:g} L | {r.current_stock}{flag} | {_rupees(r.selling_price)} | {r.shelf_location} |")
    return "\n".join(lines)


def tool_get_urgent_restock() -> str:
    df = compute_sales_forecast(30)
    urgent = df[df["reorder_urgency"].isin(["CRITICAL_ORDER_NOW", "ORDER_SOON"])]
    if urgent.empty:
        return "Nothing needs ordering right now — every product lasts longer than its supplier's delivery time."
    lines = [f"{len(urgent)} item{'s' if len(urgent) != 1 else ''} should be ordered:", "",
             "| Product | Left | Lasts about | Order | When | Supplier |", "|---|---|---|---|---|---|"]
    for r in urgent.head(10).itertuples(index=False):
        d = max(1, round(r.days_to_stockout))
        lines.append(
            f"| {r.shade_name} {r.pack_size_litres:g} L | {r.current_stock} | {d} day{'s' if d != 1 else ''} | "
            f"{r.suggested_reorder_qty} | {r.suggested_order_date} | {r.supplier_name} ({int(r.lead_time_days)} days) |"
        )
    season = get_season_status()
    if season["active"]:
        ev = season["active"][0]
        lines.append(f"\nQuantities include {ev['event_name']} demand until it ends on {ev['end_date']}.")
    return "\n".join(lines)


def tool_get_active_alerts() -> str:
    items = flatten_alerts(generate_smart_alerts())
    if not items:
        return "No alerts — everything looks fine."
    label = {"act_now": "Act now", "soon": "Soon", "look_into": "Look into"}
    lines = [f"{len(items)} open alert{'s' if len(items) != 1 else ''}:"]
    for a in items[:8]:
        lines.append(f"- **{label[a['level']]}** — {a['title']}. {a['description']} → *{a['action']}*")
    if len(items) > 8:
        lines.append(f"\n…and {len(items) - 8} more on the Alerts screen.")
    return "\n".join(lines)


def tool_get_store_kpis() -> str:
    s = get_sales_summary(30)
    conv = f"{s['today_conversion_pct']:.1f}%" if s["today_conversion_pct"] is not None else "not available"
    change = (
        f" ({'+' if s['revenue_change_pct'] >= 0 else ''}{s['revenue_change_pct']:.0f}% vs this time yesterday)"
        if s["revenue_change_pct"] is not None else ""
    )
    return (
        f"**Today ({s['date']})**\n"
        f"- Sales: {_rupees(s['today_revenue'])}{change}\n"
        f"- Cans sold: {s['today_units']} across {s['today_invoices']} bills\n"
        f"- Visitors: {s['today_visitors']} — {conv} bought something\n"
        f"- Last 30 days: {_rupees(s['period_revenue'])}\n"
        f"- Stock value: {_rupees(s['total_inventory_cost'])} at cost ({s['total_active_skus']} products)\n"
        f"- Items at or below minimum stock: {s['low_stock_count']}"
    )


def tool_recommend_colour(query: str) -> str:
    shades = search_shades_by_query(query, limit=4)
    if not shades:
        return f"No shades match “{query}”. Try a room (*bedroom*) or a colour (*beige*, *blue*)."
    lines = [f"Shades for “{query}”:"]
    for s in shades:
        status = f"in stock ({s['total_stock']} cans)" if s["in_stock"] else "out of stock"
        lines.append(f"- **{s['brand']} {s['shade_name']}** ({s['shade_code']}, {s['finish']}) — {status}")
        if not s["in_stock"] and s.get("alternatives"):
            alt = s["alternatives"][0]
            lines.append(f"  - Closest in stock: {alt['brand']} {alt['shade_name']} ({alt['current_stock']} cans)")
    lines.append("\nOpen **Colour Studio** to preview any of these on the customer's wall.")
    return "\n".join(lines)


def tool_camera_summary(topic: str = "all") -> str:
    """What the shop camera saw: buyers, shelves, fast movers, theft."""
    from src import shop_insights as si
    from src.formatting import rupees

    src = si.best_source()
    note = "" if src == "camera" else " *(sample camera history)*"
    out = []
    if topic in ("all", "buyers"):
        b = si.buyers(src)
        line = f"**Visitors today{note}:** {b['visitors']} came in"
        if b["bills"] is not None and b["visitors"]:
            line += f", {b['bills']} bills — **{b['bought_pct']:.0f}% bought**"
        line += f"; {b['left_without_counter']} of {b['finished_visits']} finished visits never reached the counter."
        out.append(line)
        for w in b["weak_hours"][:2]:
            out.append(f"- {w['label']}: {w['visitors']} came in, only {w['bills']} bought.")
    if topic in ("all", "shelves"):
        po = si.preorder(src)
        if po["refill"]:
            out.append("**Refill now:** " + " ".join(r["message"] for r in po["refill"][:3]))
        else:
            out.append("**Shelves:** every watched shelf is stocked.")
        watched = [o for o in po["order"] if o["watched"]][:3]
        for o in watched:
            out.append(f"- Pre-order **{o['label']}**: {o['qty']} cans — {o['when'].lower()}. {o['message']}")
    if topic in ("all", "movers"):
        fm = si.fast_movers(src, 7)
        top = fm["top"].head(3)
        if not top.empty:
            out.append("**Fastest movers this week:** " + ", ".join(f"{r.label} ({int(r.units)} cans)" for r in top.itertuples()))
        for h in fm["hesitant"][:1]:
            out.append(f"**Picked up but not bought:** {h['label']} — {h['message']}")
    if topic in ("all", "theft"):
        th = si.theft(src)
        out.append(f"**Theft checks:** {th['walkouts_today']} walk-out(s) with a can today, {th['open']} alert(s) to check this week"
                   + (f", shelf-vs-till gap {rupees(th['unbilled_value'])} today." if th["unbilled_value"] else "."))
    return "\n".join(out)


def _deal_help() -> str:
    return (
        "Use **Stock & Reorder → Deal Checker** for supplier offers. Enter the product, the offered price per can, "
        "how many cans, and months until expiry — it tells you whether it will sell before it expires and the profit."
    )


def deterministic_assistant_response(user_message: str) -> str:
    """Offline answers: route the question to the right data tool."""
    msg = user_message.lower()
    if any(k in msg for k in ("clearance", "deal", "supplier offer", "discount")):
        return _deal_help()
    if any(k in msg for k in ("running low", "run out", "running out", "low stock", "low on", "reorder", "order", "restock")) \
            or re.search(r"\blow\b", msg):
        return tool_get_urgent_restock()
    if any(k in msg for k in ("theft", "steal", "stolen", "shoplift", "walk out", "walk-out", "unpaid", "security")):
        return tool_camera_summary("theft")
    if any(k in msg for k in ("shelf", "shelves", "refill")):
        return tool_camera_summary("shelves")
    if any(k in msg for k in ("fast mov", "fast-mov", "best sell", "moving", "picked up", "put back")):
        return tool_camera_summary("movers")
    if any(k in msg for k in ("bought", "buying", "customers", "conversion", "footfall", "visitor", "camera")):
        return tool_camera_summary("buyers")
    if any(k in msg for k in ("alert", "expir", "slow", "stuck", "attention", "problem", "issue")):
        return tool_get_active_alerts()
    if any(k in msg for k in ("today", "sales", "revenue", "kpi", "visitor", "footfall", "conversion", "how did we do", "summary")):
        return tool_get_store_kpis()
    if any(k in msg for k in ("colour", "color", "shade for", "suggest", "recommend", "bedroom", "living room", "kitchen", "wall")):
        return tool_recommend_colour(user_message)
    if any(k in msg for k in ("stock", "have", "available", "cans of", "how many")):
        return tool_check_stock(user_message)
    return (
        "I can help with:\n"
        "- **Stock** — *Do we have Royale Blue?*\n"
        "- **Reorders** — *What should I order today?*\n"
        "- **Alerts** — *What needs attention?*\n"
        "- **Today's numbers** — *How did we do today?*\n"
        "- **Camera** — *How many customers bought today? Which shelf needs refilling? Any theft?*\n"
        "- **Colours** — *Suggest a calm bedroom colour*"
    )


def _live_context() -> str:
    season = get_season_status()
    season_txt = ", ".join(f"{e['event_name']} ({e['start_date']} to {e['end_date']}, {e['expected_multiplier']}x {e['impact_category']})"
                           for e in season["active"]) or "none"
    return (
        f"[TODAY]\n{tool_get_store_kpis()}\n\n[ACTIVE SEASON]\n{season_txt}\n\n"
        f"[REORDER PLAN]\n{tool_get_urgent_restock()}\n\n[ALERTS]\n{tool_get_active_alerts()}\n\n"
        f"[SHOP CAMERA]\n{_safe_camera()}"
    )


def _safe_camera() -> str:
    try:
        return tool_camera_summary("all")
    except Exception:  # noqa: BLE001
        return "not available"


def answer(user_message: str, chat_history: Optional[List[Dict[str, str]]] = None,
           api_key: Optional[str] = None) -> Dict[str, Any]:
    """Returns {"text", "mode"} where mode is 'gemini', 'offline' or 'offline_fallback'."""
    key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if key and GENAI_AVAILABLE:
        try:
            client = genai.Client(api_key=key)
            history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in (chat_history or [])[-6:])
            prompt = f"{_live_context()}\n\n[CONVERSATION SO FAR]\n{history}\n\nUSER: {user_message}"
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            )
            if resp and resp.text:
                return {"text": resp.text, "mode": "gemini"}
        except Exception:
            text = deterministic_assistant_response(user_message)
            return {"text": "*Gemini didn't respond, so this is the offline answer.*\n\n" + text, "mode": "offline_fallback"}
    return {"text": deterministic_assistant_response(user_message), "mode": "offline"}


def chat_with_assistant(user_message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Backwards-compatible wrapper returning just the text."""
    return answer(user_message, chat_history)["text"]
