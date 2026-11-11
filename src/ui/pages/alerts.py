"""Alerts — what needs attention right now."""
from datetime import datetime
from html import escape

import streamlit as st

from src import clock, database as db
from src.dashboard_views import render_hourly_traffic_chart
from src.ui import data
from src.ui.theme import chip, html, page_header, show_chart

GROUPS = ["All", "Running out", "Lost sales", "Shelves", "Theft", "Expiring", "Money stuck", "Visitors vs bills"]
HIDE_DAYS = 3


def _act(alert: dict, action: str, message: str) -> None:
    db.record_alert_action(alert["id"], action, HIDE_DAYS)
    data.clear_all()
    st.toast(message)
    st.rerun()


def _card(alert: dict, idx: int) -> None:
    with st.container(border=True):
        html(f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
             f'{chip(alert["level"])}<span class="apx-muted" style="font-size:12px">{escape(alert["group"])}</span></div>')
        html(f'<div style="font-size:15px;font-weight:600;line-height:1.35;margin:8px 0 4px">{escape(alert["title"])}</div>'
             f'<div class="apx-muted" style="line-height:1.5">{escape(alert["description"])}</div>')
        b1, b2 = st.columns([3, 2])
        t = alert["type"]
        if t == "LOW_STOCK":
            qty = alert.get("detail", {}).get("qty", 0)
            if b1.button(f"Mark {qty} cans ordered", key=f"al_a_{idx}", type="primary", use_container_width=True):
                db.mark_ordered(alert["product_id"], qty)
                _act(alert, "ordered", f"Noted: {qty} cans on order.")
        elif t in ("EXPIRING_BATCH", "EXPIRED_BATCH"):
            if b1.button(alert["action"], key=f"al_a_{idx}", type="primary", use_container_width=True):
                _act(alert, "offer", f"{alert['action']} — noted. Hidden for {HIDE_DAYS} days.")
        elif t == "REFILL_SHELF" and alert.get("move"):
            if b1.button(alert["action"], key=f"al_a_{idx}", type="primary", use_container_width=True):
                db.log_shelf_event(alert["source"], alert["slot_id"], alert.get("product_id") or None, "restock",
                                   alert["move"])
                _act(alert, "refilled", f"Shelf refilled with {alert['move']} cans.")
        elif t == "UNPAID_EXIT":
            if b1.button("Checked — real", key=f"al_a_{idx}", type="primary", use_container_width=True):
                db.update_security_event_status(alert["event_id"], "SEEN")
                _act(alert, "checked", "Marked as checked.")
            if b2.button("False alarm", key=f"al_f_{idx}", use_container_width=True):
                db.update_security_event_status(alert["event_id"], "FALSE_ALARM")
                _act(alert, "false_alarm", "Marked as a false alarm.")
            return
        elif t == "CONVERSION_DROP":
            with b1.popover(alert["action"], use_container_width=True):
                day = datetime.strptime(alert["day"], "%Y-%m-%d").date()
                html(f"<b>Visitors vs bills · {clock.fmt_day(day)}</b>")
                show_chart(render_hourly_traffic_chart(data.hourly_traffic(alert["day"])), key=f"al_chart_{idx}")
        else:
            if b1.button(alert["action"], key=f"al_a_{idx}", type="primary", use_container_width=True):
                _act(alert, "acted", f"{alert['action']} — noted.")
        if b2.button("Mark done", key=f"al_d_{idx}", use_container_width=True):
            _act(alert, "done", f"Done. Hidden for {HIDE_DAYS} days.")


def render() -> None:
    page_header("Alerts", "Each alert says what's wrong, why it matters, and the one thing to do. Mark it done when handled.")
    day = clock.today_str()
    alerts = data.open_alerts(day)
    counts = {g: sum(1 for a in alerts if a["group"] == g) for g in GROUPS[1:]}
    counts["All"] = len(alerts)
    group = st.radio("Group", GROUPS, format_func=lambda g: f"{g} · {counts.get(g, 0)}", horizontal=True,
                     key="al_group", label_visibility="collapsed")
    shown = [a for a in alerts if group == "All" or a["group"] == group]
    if not shown:
        with st.container(border=True):
            html(f'{chip("healthy", "All clear")} <span class="apx-muted">Nothing needs attention in this group.</span>')
        return
    cols_per_row = 3
    for start in range(0, len(shown), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (i, a) in zip(cols, list(enumerate(shown))[start:start + cols_per_row]):
            with col:
                _card(a, i)
    st.caption(f"“Mark done” hides an alert for {HIDE_DAYS} days. If the problem is still there after that, it comes back.")
