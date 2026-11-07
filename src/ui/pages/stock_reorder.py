"""Stock & Reorder — what to order, how much, and by when. Includes the Deal Checker."""
from html import escape

import pandas as pd
import streamlit as st

from src import clock, database as db
from src.formatting import pack, rupees, rupees_short
from src.predictor import evaluate_supplier_clearance_deal
from src.ui import data
from src.ui.theme import chip, html, kpi_card, page_header, show_table

WHEN_STYLE = {
    "CRITICAL_ORDER_NOW": ("#B42318", "#FCEBE9"),
    "ORDER_SOON": ("#B54708", "#FDF1E2"),
    "ON_ORDER": ("#1D4ED8", "#E7EDFB"),
    "PLANNED": ("#15202B", "#F0F1F4"),
    "SUFFICIENT_STOCK": ("#5A6472", "#FFFFFF"),
}
STATUS_WORDS = {
    "OUT_OF_STOCK": "Out of stock", "EXPIRED": "Expired", "LOW_STOCK": "Low", "EXPIRING_SOON": "Expiring soon",
    "OVERSTOCK": "Too much", "HEALTHY": "Healthy",
}


def _reorder_plan(day: str) -> None:
    fc = data.forecast(day)
    now = fc[fc["reorder_urgency"] == "CRITICAL_ORDER_NOW"]
    soon = fc[fc["reorder_urgency"] == "ORDER_SOON"]
    ordered = fc[fc["reorder_urgency"] == "ON_ORDER"]
    planned = fc[fc["reorder_urgency"] == "PLANNED"]
    to_buy = pd.concat([now, soon])

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Order today", f"{len(now)} item{'s' if len(now) != 1 else ''}",
                 "Will run out before the supplier delivers", attention=len(now) > 0,
                 value_colour="#B42318" if len(now) else None)
    with c2:
        kpi_card("Order within a few days", f"{len(soon)} item{'s' if len(soon) != 1 else ''}", "A little buffer left",
                 value_colour="#B54708" if len(soon) else None)
    with c3:
        kpi_card("Cost of these orders", rupees_short(to_buy["estimated_order_cost"].sum()), "At supplier cost price")

    st.write("")
    groups = {
        f"Order today · {len(now)}": now,
        f"Within a few days · {len(soon)}": soon,
        f"Already ordered · {len(ordered)}": ordered,
        f"Planned · {len(planned)}": planned,
    }
    pick = st.radio("Show", list(groups), horizontal=True, key="sr_group", label_visibility="collapsed")
    rows = groups[pick]
    if rows.empty:
        st.caption("Nothing in this group.")
        return

    table = pd.DataFrame({
        "Product": rows["shade_name"] + " · " + rows["brand"],
        "Pack": rows["pack_size_litres"].map(pack),
        "On shelf": rows["current_stock"],
        "Coming": rows["on_order"],
        "Sells / day": rows["effective_velocity"].round(1),
        "Lasts about": rows["days_to_stockout"].map(lambda d: "under a day" if d < 1 else f"{d:.0f} days"),
        "Supplier": rows["supplier_name"] + " · " + rows["lead_time_days"].astype(int).astype(str) + " days",
        "Order": rows["suggested_reorder_qty"],
        "When": rows["suggested_order_date"],
        "_u": rows["reorder_urgency"],
    })

    urgencies = table["_u"].tolist()

    def style_when(col):
        return [f"color:{WHEN_STYLE[u][0]};background-color:{WHEN_STYLE[u][1]};font-weight:600" for u in urgencies]

    styled = table.drop(columns="_u").style.apply(style_when, subset=["When"])
    show_table(styled)
    st.caption(
        "Order = next 30 days' demand (with season effects day by day) + 5 days' safety − on shelf − already coming, "
        "rounded up to the supplier's minimum order."
    )

    csv = table.drop(columns="_u").to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns([1, 1])
    c1.download_button("Download this list (CSV)", csv, file_name=f"order_list_{day}.csv", mime="text/csv",
                       use_container_width=True)
    if pick.startswith(("Order today", "Within")):
        if c2.button(f"Mark all {len(rows)} as ordered", key="sr_mark_all", use_container_width=True):
            for r in rows.itertuples(index=False):
                db.mark_ordered(int(r.product_id), int(r.suggested_reorder_qty))
            data.clear_all()
            st.toast(f"Marked {len(rows)} items as ordered.")
            st.rerun()


def _all_stock(day: str) -> None:
    inv = data.inventory(day)
    c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
    q = c1.text_input("Search", placeholder="Shade, brand or SKU", key="sr_q")
    counts = inv["health_status"].value_counts().to_dict()
    options = ["All"] + [k for k in STATUS_WORDS if counts.get(k)]
    labels = {k: f"{STATUS_WORDS.get(k, k)} · {counts.get(k, len(inv))}" if k != "All" else f"All · {len(inv)}" for k in options}
    status = c2.radio("Status", options, format_func=lambda k: labels[k], horizontal=True, key="sr_status",
                      label_visibility="collapsed")
    view = inv
    if status != "All":
        view = view[view["health_status"] == status]
    if q:
        ql = q.lower()
        hay = (view["brand"] + " " + view["shade_name"] + " " + view["sku"]).str.lower()
        view = view[hay.str.contains(ql, regex=False)]
    show_table(pd.DataFrame({
        "SKU": view["sku"], "Product": view["shade_name"] + " · " + view["brand"], "Category": view["category"],
        "Pack": view["pack_size_litres"].map(pack), "On shelf": view["current_stock"], "Coming": view["on_order"],
        "Price": view["selling_price"].map(rupees), "Expires in": view["days_to_expiry"].map(lambda d: f"{d} days" if d >= 0 else "expired"),
        "Status": view["health_status"].map(STATUS_WORDS), "Shelf": view["shelf_location"], "Supplier": view["supplier_name"],
    }))


def _deal_checker(day: str) -> None:
    inv = data.inventory(day)
    options = inv.sort_values(["brand", "shade_name", "pack_size_litres"])
    labels = {int(r.product_id): f"{r.brand} {r.shade_name} · {pack(r.pack_size_litres)} (usual cost {rupees(r.cost_price)})"
              for r in options.itertuples(index=False)}
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        with st.form("deal_form", border=True):
            html('<div class="apx-step-t">Supplier offer</div>')
            pid = st.selectbox("Product", list(labels), format_func=lambda k: labels[k])
            usual = float(options.loc[options["product_id"] == pid, "cost_price"].iloc[0])
            c1, c2 = st.columns(2)
            price = c1.number_input("Offered price per can (₹)", min_value=1.0, value=round(usual * 0.8), step=10.0)
            qty = c2.number_input("Number of cans", min_value=1, value=40, step=5)
            months = st.number_input("Months until expiry", min_value=1, value=12, step=1)
            go = st.form_submit_button("Check this deal", type="primary", use_container_width=True)
    with right:
        if go or SS_DEAL in st.session_state:
            if go:
                st.session_state[SS_DEAL] = evaluate_supplier_clearance_deal(int(pid), float(price), int(qty), int(months))
            r = st.session_state[SS_DEAL]
            if "error" in r:
                st.error(r["error"])
                return
            level = {"success": "healthy", "warning": "soon", "critical": "act_now"}[r["badge"]]
            with st.container(border=True):
                html(chip(level, r["recommendation"].capitalize()))
                html(f'<p style="font-size:15px;margin:10px 0">{escape(r["verdict"])}</p>')
                m1, m2 = st.columns(2)
                m1.metric("Discount vs usual cost", f"{r['discount_pct']:.0f}%")
                m2.metric("Money needed now", rupees(r["total_investment"]))
                m3, m4 = st.columns(2)
                m3.metric("Profit after holding cost", rupees(r["net_projected_profit"]))
                m4.metric("Sells in about", f"{r['months_to_sell']:.1f} months")
                html(f'<div class="apx-muted">{escape(r["action"])}</div>')
        else:
            st.caption("Enter the offer on the left to see whether it will sell before it expires, and the profit.")


SS_DEAL = "sr_deal_result"


def render() -> None:
    page_header("Stock & Reorder", "What to order, how much, and by when — with supplier delivery times built in.")
    day = clock.today_str()
    t1, t2, t3 = st.tabs(["Reorder plan", "All stock", "Deal Checker"])
    with t1:
        _reorder_plan(day)
    with t2:
        _all_stock(day)
    with t3:
        _deal_checker(day)
