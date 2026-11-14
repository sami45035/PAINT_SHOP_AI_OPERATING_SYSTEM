"""Overview — how is the shop doing today?"""
from html import escape

import streamlit as st

from src import clock
from src.dashboard_views import render_hourly_traffic_chart, render_sales_trend_chart
from src.formatting import rupees, rupees_short
from src.ui import data, nav
from src.ui.theme import banner, change_html, chip, html, kpi_card, page_header, show_chart, swatch_row  # noqa: F401


def _season_banner(day: str) -> None:
    s = data.season(day)
    nxt = s.get("next")
    nxt_txt = ""
    if nxt:
        start = clock.fmt_day(nxt["start"]) if hasattr(nxt["start"], "strftime") else nxt["start_date"]
        nxt_txt = f" Next: {escape(nxt['event_name'])} from {start}."
    if s["active"]:
        ev = s["active"][0]
        end = clock.fmt_day(ev["end"])
        banner("info", "Season",
               f"<b>{escape(ev['event_name'])}</b> (until {end}) — {escape(ev['impact_category'])} demand about "
               f"<b>{ev['expected_multiplier']:g}×</b> normal.{nxt_txt}")
    elif nxt:
        banner("info", "Season", f"No festival or season effect today.{nxt_txt}")


def _traffic_insight(hourly) -> str:
    busy = hourly[hourly["visitors"] >= 5]
    if busy.empty:
        return ""
    peak = hourly.loc[hourly["visitors"].idxmax()]
    worst = busy.loc[busy["conversion_pct"].idxmin()]
    label = lambda h: f"{(int(h) % 12) or 12} {'AM' if h < 12 else 'PM'}"
    txt = f"Busiest at {label(peak['hour'])} ({int(peak['visitors'])} visitors)."
    if worst["conversion_pct"] < 25:
        txt += f" Only {worst['conversion_pct']:.0f}% bought at {label(worst['hour'])} — check stock and the billing queue then."
    return txt


def _camera_strip(day: str) -> None:
    """What the shop camera adds: who bought, shelves to refill, theft checks."""
    from src import shop_insights as si

    try:
        src = si.best_source()
        b = data.cam_buyers(src, day)
        po = data.cam_preorder(src, day)
        th = data.cam_theft(src, day)
        fm = data.cam_fast_movers(src, day)
    except Exception:  # noqa: BLE001 - camera tables missing
        return
    with st.container(border=True):
        tag = "" if src == "camera" else ' <span class="apx-muted" style="font-size:12px">· sample history</span>'
        html(f'<div class="apx-step-t" style="font-size:18px">From the shop camera{tag}</div>')
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Left without the counter", str(b["left_without_counter"]),
                     f"of {b['finished_visits']} visits today")
        with c2:
            hes = fm["hesitant"][0]["label"] if fm["hesitant"] else "—"
            kpi_card("Picked up, not bought", hes, "most put back this week")
        with c3:
            low = [r for r in po["refill"]]
            kpi_card("Shelves to refill", str(len(low)), ", ".join(r["slot_name"] for r in low[:3]) or "all stocked",
                     attention=bool(low))
        with c4:
            kpi_card("Theft checks", str(th["open"]), f"walk-outs today: {th['walkouts_today']}",
                     attention=th["open"] > 0, value_colour="#B42318" if th["open"] else None)
        nav.link("camera", "Open Store Camera →")


def render() -> None:
    day = clock.today_str()
    now = clock.now()
    page_header("Overview", f"{now:%A}, {now.day} {now:%B %Y}")
    _season_banner(day)

    s = data.sales_summary(day)
    alerts = data.open_alerts(day)
    urgent = [a for a in alerts if a["level"] == "act_now"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Today's sales", rupees(s["today_revenue"]), change_html(s["revenue_change_pct"]))
    with c2:
        kpi_card("Cans sold", str(s["today_units"]), f"across {s['today_invoices']} bills")
    with c3:
        conv = s["today_conversion_pct"]
        kpi_card("Visitors who bought", f"{conv:.1f}%" if conv is not None else "—",
                 f"{s['today_invoices']} bills ÷ {s['today_visitors']} visitors" if s["today_visitors"] else "No visitor count yet")
    with c4:
        kpi_card("Stock value (cost)", rupees_short(s["total_inventory_cost"]), f"{s['total_active_skus']} products on shelf")
    with c5:
        kpi_card("Needs attention", str(len(alerts)), f"{len(urgent)} to act on now", attention=bool(urgent),
                 value_colour="#B42318" if urgent else None)

    st.write("")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            html('<div class="apx-step-t" style="font-size:18px">Daily sales · last 45 days</div>')
            show_chart(render_sales_trend_chart(data.daily_sales(day)), key="sales45")
    with right:
        with st.container(border=True):
            html('<div class="apx-step-t" style="font-size:18px">Visitors vs bills · today by hour</div>')
            hourly = data.hourly_traffic(day)
            show_chart(render_hourly_traffic_chart(hourly), key="hourly")
            insight = _traffic_insight(hourly)
            if insight:
                html(f'<div class="apx-muted">{escape(insight)}</div>')

    _camera_strip(day)

    a, b, c = st.columns(3)
    ts = data.top_and_slow(day)
    with a:
        with st.container(border=True):
            html('<div class="apx-step-t" style="font-size:18px">Best sellers · 90 days</div>')
            rows = "".join(
                swatch_row(r.hex_code, r.shade_name, r.brand, f"{int(r.total_units_sold)} cans")
                for r in ts["top_shades"].head(5).itertuples(index=False)
            )
            html(rows)
    with b:
        with st.container(border=True):
            html('<div class="apx-step-t" style="font-size:18px">Money stuck in slow sellers</div>')
            stuck = [x for x in alerts if x["type"] == "DEAD_STOCK"][:4]
            if stuck:
                html("".join(
                    swatch_row(x["hex_code"], x["title"].split(" — ")[0], x["description"],
                               f'<span style="color:#B54708">{escape(x["title"].split(" — ")[1].split(" sitting")[0])}</span>')
                    for x in stuck
                ))
            else:
                html('<div class="apx-muted">Nothing stuck — every product sold in the last 60 days.</div>')
    with c:
        with st.container(border=True):
            html('<div class="apx-step-t" style="font-size:18px">Do today</div>')
            if not alerts:
                html('<div class="apx-muted">All clear.</div>')
            for x in alerts[:4]:
                html(f'<div class="apx-row" style="align-items:flex-start">{chip(x["level"])}'
                     f'<span class="apx-row-main">{escape(x["title"])}</span></div>')
            nav.link("alerts", f"See all {len(alerts)} alerts →")
