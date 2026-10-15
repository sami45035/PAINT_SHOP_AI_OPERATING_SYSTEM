"""
Conversational AI Shopkeeper Copilot (Layer 7)
Provides natural-language store management with Gemini API integration
and an intelligent local deterministic fallback when an API key is not yet set.
"""
import os
import json
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from src.config import GEMINI_API_KEY, GEMINI_MODEL, DB_PATH
from src.database import get_sales_summary, get_inventory_status, get_footfall_pos_fusion, get_recent_security_events
from src.predictor import compute_sales_forecast, evaluate_supplier_clearance_deal
from src.smart_alerts import generate_smart_alerts
from src.colour_assistant import search_shades_by_query

# Try importing google-genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

SYSTEM_INSTRUCTION = """
You are 'Aura', the intelligent AI Shop Assistant for an independent Paint & Hardware Store.
You have access to live shop databases: products, inventory levels, sales velocity, supplier lead times,
footfall foot-traffic counters, security alerts, and color catalogs.
Your goals:
1. Help the shopkeeper make fast, data-driven decisions on purchasing and inventory.
2. Prevent stockouts of high-demand paints (especially ahead of festivals like Diwali).
3. Warn of expiring batches and suggest promotional liquidation discounts.
4. Help customers find shades, in-stock sizes, and nearest alternatives.
5. Keep your tone professional, concise, proactive, and commercially sharp.
"""

# Tool functions available to the assistant
def tool_check_stock(query: str) -> str:
    df = get_inventory_status()
    q = query.lower()
    matches = df[
        df["shade_name"].str.lower().str.contains(q) |
        df["brand"].str.lower().str.contains(q) |
        df["sku"].str.lower().str.contains(q) |
        df["category"].str.lower().str.contains(q)
    ]
    if matches.empty:
        return f"No products matching '{query}' found in store inventory."
    
    lines = [f"Found {len(matches)} matching SKUs:"]
    for _, r in matches.head(8).iterrows():
        status_emoji = "🟢" if r["current_stock"] > r["min_threshold"] else ("🔴" if r["current_stock"] <= 2 else "🟡")
        lines.append(f"- {status_emoji} **{r['brand']} {r['shade_name']}** ({r['pack_size_litres']}L) | Stock: **{r['current_stock']} cans** | Price: ₹{r['selling_price']} | Shelf: {r['shelf_location']} (Status: {r['health_status']})")
    return "\n".join(lines)

def tool_get_urgent_restock() -> str:
    df = compute_sales_forecast(30)
    urgent = df[df["reorder_urgency"].isin(["CRITICAL_ORDER_NOW", "ORDER_SOON"])]
    if urgent.empty:
        return "All paint inventory levels are currently healthy! No urgent reorders required today."
    
    lines = [f"⚠️ **{len(urgent)} Products Require Immediate Reorder:**"]
    for _, r in urgent.head(6).iterrows():
        lines.append(
            f"- 🔴 **{r['brand']} {r['shade_name']}** ({r['pack_size_litres']}L): "
            f"Stock: **{r['current_stock']} left** | Days to Stockout: **{r['days_to_stockout']}d** | "
            f"Supplier Lead Time: **{r['lead_time_days']}d** | "
            f"Suggested Order: **{r['suggested_reorder_qty']} units** (Supplier: {r['supplier_name']})"
        )
    return "\n".join(lines)

def tool_get_active_alerts() -> str:
    alerts = generate_smart_alerts()
    crit_count = len(alerts["critical"])
    exp_count = len(alerts["expiring"])
    dead_count = len(alerts["dead_stock"])
    anom_count = len(alerts["anomalies"])
    
    lines = [f"📋 **Active Store Alerts Summary ({crit_count + exp_count + dead_count + anom_count} total):**"]
    
    if alerts["critical"]:
        lines.append("\n**🚨 Critical Low Stock:**")
        for a in alerts["critical"][:3]:
            lines.append(f"- {a['title']}: {a['description']} -> *Action:* {a['action']}")
            
    if alerts["expiring"]:
        lines.append("\n**⏳ Expiring Batches (< 45 Days):**")
        for a in alerts["expiring"][:3]:
            lines.append(f"- {a['title']}: {a['description']} -> *Action:* {a['action']}")
            
    if alerts["dead_stock"]:
        lines.append("\n**📦 Dead Capital / Stagnant Inventory:**")
        for a in alerts["dead_stock"][:2]:
            lines.append(f"- {a['title']}: {a['description']} -> *Remediation:* {a['action']}")
            
    return "\n".join(lines)

def tool_get_store_kpis() -> str:
    summary = get_sales_summary(30)
    fusion_df = get_footfall_pos_fusion(7)
    avg_conv = round(fusion_df["conversion_rate_pct"].mean(), 1) if not fusion_df.empty else 34.0
    
    return (
        f"📊 **Store Performance Snapshot (Today: {summary['max_date']}):**\n"
        f"- **Today's Revenue:** ₹{summary['today_revenue']:,.2f} ({summary['today_units']} cans sold across {summary['today_invoices']} invoices)\n"
        f"- **30-Day Total Revenue:** ₹{summary['period_revenue']:,.2f}\n"
        f"- **Active SKUs in Store:** {summary['total_active_skus']} products\n"
        f"- **Total Inventory Valuation:** ₹{summary['total_inventory_cost']:,.2f} cost (₹{summary['total_inventory_retail']:,.2f} retail)\n"
        f"- **Store Footfall Conversion Rate:** ~**{avg_conv}%** (Visitors $\\rightarrow$ Buyers)\n"
        f"- **Critical Low-Stock SKUs:** {summary['low_stock_count']} items"
    )

def tool_recommend_colour(query: str) -> str:
    shades = search_shades_by_query(query, limit=3)
    if not shades:
        return f"No specific shades found matching '{query}'. Try searching by room ('living room') or color ('beige', 'emerald')."
    
    lines = [f"🎨 **Recommended Color Matches for '{query}':**"]
    for s in shades:
        stock_status = "In Stock" if s["in_stock"] else "Out of Stock"
        lines.append(
            f"- **{s['brand']} {s['shade_name']}** (`{s['shade_code']}`) - {s['finish']}\n"
            f"  Hex: `{s['hex_code']}` | Category: {s['category']} | Status: **{stock_status}** ({s['total_stock']} units)"
        )
        if not s["in_stock"] and s.get("alternatives"):
            alt = s["alternatives"][0]
            lines.append(f"  *Suggested in-stock alternative:* {alt['brand']} {alt['shade_name']} ({alt['current_stock']} units available)")
    return "\n".join(lines)

def deterministic_assistant_response(user_message: str) -> str:
    """
    Intelligent offline fallback reasoning engine.
    Interprets user intent and routes to relevant database tools.
    """
    msg = user_message.lower()

    if any(k in msg for k in ["stock", "available", "have", "inventory", "check", "cans"]):
        if any(k in msg for k in ["low", "out of", "run out", "empty", "critical"]):
            return tool_get_urgent_restock()
        # Extract query
        words = user_message.split()
        keywords = [w for w in words if w.lower() not in ["do", "we", "have", "in", "stock", "check", "for", "is", "the", "are", "there", "any"]]
        sub_query = " ".join(keywords) if keywords else "White"
        return tool_check_stock(sub_query)

    elif any(k in msg for k in ["order", "reorder", "purchase", "buy", "stockup", "restock"]):
        return tool_get_urgent_restock()

    elif any(k in msg for k in ["alert", "expir", "dead", "warning", "issue", "attention"]):
        return tool_get_active_alerts()

    elif any(k in msg for k in ["kpi", "performance", "revenue", "sales", "footfall", "conversion", "today", "summary"]):
        return tool_get_store_kpis()

    elif any(k in msg for k in ["color", "colour", "shade", "paint", "bedroom", "living", "room", "recommend", "wall"]):
        return tool_recommend_colour(user_message)

    elif any(k in msg for k in ["clearance", "deal", "discount", "supplier offer"]):
        return (
            "🏷️ **Supplier Clearance Deal Evaluator:**\n"
            "To evaluate a supplier bulk discount, tell me:\n"
            "1. Product SKU or Name\n"
            "2. Offered discounted cost per unit\n"
            "3. Number of units offered\n"
            "4. Remaining shelf life (months)\n\n"
            "Example: *'Evaluate supplier offer for Asian Paints Royale Blue 10L at ₹420 for 30 units with 10 months shelf life.'*"
        )
    else:
        return (
            f"Hello! I am Aura, your **Paint Shop AI Operating System** copilot.\n\n"
            f"I can assist you with:\n"
            f"- **📦 Inventory Checks:** *'Do we have Royale Blue in stock?'*\n"
            f"- **📈 Restock Recommendations:** *'What paints should I order today?'*\n"
            f"- **⚠️ Store Alerts:** *'Show me expiring batches and low stock alerts'*\n"
            f"- **📊 Performance KPIs:** *'What was today's sales and footfall conversion?'*\n"
            f"- **🎨 Customer Colour Advice:** *'Suggest a warm beige for living room'*\n"
            f"- **🏷️ Clearance Evaluator:** *'Should I buy discounted stock from supplier?'*\n\n"
            f"What would you like to check?"
        )

def chat_with_assistant(user_message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Primary interface for the AI Shopkeeper Assistant.
    Uses Gemini API if key is available; otherwise uses deterministic reasoning.
    """
    api_key = os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    
    if api_key and GENAI_AVAILABLE:
        try:
            client = genai.Client(api_key=api_key)
            # Compile live shop context into the prompt
            context = f"""
[LIVE SHOP DATA CONTEXT]
{tool_get_store_kpis()}

[URGENT RESTOCK STATUS]
{tool_get_urgent_restock()}

[ACTIVE ALERTS]
{tool_get_active_alerts()}
"""
            full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{context}\n\nUser Question: {user_message}"
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            # If API fails or hits quota, gracefully fall back
            fallback_text = deterministic_assistant_response(user_message)
            return f"*(Gemini live response fallback - {str(e)[:45]}...)*\n\n{fallback_text}"

    # Deterministic local fallback
    return deterministic_assistant_response(user_message)
