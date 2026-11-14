"""Sales Forecast — what will sell in the next 30 days, learned from the shop's own sales."""
from datetime import timedelta
from html import escape

import numpy as np
import streamlit as st

from src import clock
from src import forecast_charts as fch
from src.formatting import rupees, rupees_short
from src.ui import data, nav
from src.ui.theme import chip, html, kpi_card, page_header, show_chart, swatch_row

SS = st.session_state
SHORT = {"Diwali Pre-Festival Surge": "Diwali", "Post-Monsoon Waterproofing": "Post-monsoon",
         "Winter Wedding Season": "Wedding season", "Monsoon Exterior Lull": "Monsoon lull"}


def _title(text: str, sub: str = "") -> None:
    html(f'<div class="apx-step-t" style="font-size:18px">{escape(text)}</div>'
         + (f'<div class="apx-muted" style="margin:2px 0 8px">{sub}</div>' if sub else ""))


def _effect(x: float) -> str:
    return f"{x:.2f}×".replace(".00×", "×")


def render() -> None:
    day = clock.today_str()
    page_header("Sales Forecast", "What will sell in the next 30 days — learned from six months of your own sales.")
    o = data.forecast_outlook(day)
    bt = data.forecast_backtest(day)
    rb14 = data.forecast_rolling(day, 14)
    rb28 = data.forecast_rolling(day, 28)

    daily = o["daily"]
    nxt7 = daily.head(7)
    p80 = float(np.percentile(rb28["periods"]["ai_err_pct"], 80)) / 100 if len(rb28["periods"]) else 0.15
    total30 = float(daily["revenue"].sum())
    hist = o["history"]
    last30 = float(hist[hist["day"] >= o["today"] - timedelta(days=30)]["revenue"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Next 7 days", rupees_short(nxt7["revenue"].sum()), f"about {nxt7['units'].sum():.0f} cans")
    with c2:
        chg = (total30 - last30) / last30 * 100 if last30 else None
        sub = f"likely {rupees_short(total30 * (1 - p80))}–{rupees_short(total30 * (1 + p80))}"
        if chg is not None:
            sub += f' · <span class="{"apx-up" if chg >= 0 else "apx-down"}">{"▲" if chg >= 0 else "▼"} {abs(chg):.0f}%</span> vs last 30'
        kpi_card("Next 30 days", rupees_short(total30), sub)
    with c3:
        active = [e for e in o["events"] if e["start"] <= o["today"] <= e["end"] and e["days_seen"]]
        if active:
            e = max(active, key=lambda e: e["learned"])
            kpi_card("Festival now", SHORT.get(e["name"], e["name"]),
                     f"{_effect(e['learned'])} on {e['group'].lower()} (calendar said {_effect(e['calendar'])}) · ends {clock.fmt_day(e['end'])}")
        else:
            kpi_card("Festival now", "None", "normal weeks ahead")
    with c4:
        kpi_card("Forecast accuracy", f"±{rb14['ai_group_err_pct']:.0f}%",
                 f"per product group, 2 weeks ahead, tested on {rb14['windows']} past periods · "
                 f"simple average ±{rb14['simple_group_err_pct']:.0f}%")

    st.write("")
    with st.container(border=True):
        h1, h2 = st.columns([3, 1], vertical_alignment="center")
        with h1:
            _title("Sales: last 60 days and the next 30",
                   "Bars = what sold · line = AI forecast · band = likely range (from how far past forecasts were off) · shaded = festival")
        with h2:
            unit = st.radio("Show", ["Rupees", "Cans"], horizontal=True, key="fc_unit", label_visibility="collapsed")
        band_lo, band_hi = max(0.0, 1 - p80), 1 + p80
        show_chart(fch.outlook_chart(hist, daily, band_lo, band_hi, o["events"],
                                     value="revenue" if unit == "Rupees" else "units"), key="fc_outlook")
        ends = [e for e in o["events"] if o["today"] <= e["end"] <= daily["day"].max() and e["days_seen"]]
        starts = [e for e in o["events"] if o["today"] < e["start"] <= daily["day"].max()]
        notes = []
        for e in {(x["name"], x["end"]): x for x in ends}.values():
            notes.append(f"<b>{escape(SHORT.get(e['name'], e['name']))} ends {clock.fmt_day(e['end'])}</b> — "
                         f"demand for {escape(e['group'].lower())} drops back after that; don't over-order.")
        for e in {(x["name"], x["start"]): x for x in starts}.values():
            notes.append(f"<b>{escape(SHORT.get(e['name'], e['name']))} starts {clock.fmt_day(e['start'])}</b> — "
                         f"not in your history yet, so the calendar's {_effect(e['calendar'])} is counted at half strength "
                         f"({_effect(e['learned'])}) until your sales confirm it.")
        if notes:
            html('<div class="apx-muted" style="line-height:1.7">' + "<br>".join(notes) + "</div>")

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            _title("What will sell · next 30 days", "Top products by expected sales; change vs the last 30 days")
            prods = o["products"].head(10)
            rows = []
            for r in prods.to_dict("records"):
                chg = r["change_pct"]
                trend = "" if chg != chg else (
                    f'<span class="{"apx-up" if chg >= 0 else "apx-down"}">{"▲" if chg >= 0 else "▼"} {abs(chg):.0f}%</span>')
                rows.append(swatch_row(r["hex_code"], r["label"], f"{r['brand']} · {r['units']:.0f} cans expected",
                                       f"{rupees_short(r['revenue'])} {trend}"))
            html("".join(rows))
            nav.link("stock", "Order for this on Stock & Reorder →")
    with right:
        with st.container(border=True):
            _title("What the model learned from your sales")
            lr = [x for x in o["learned"] if x["days_seen"] or (x["end"] >= o["today"])]
            seen_rows = []
            for x in sorted(lr, key=lambda x: (-x["days_seen"], x["festival"])):
                status = (f'{chip("healthy", "Learned")}' if x["days_seen"] >= 7 else chip("neutral", "Calendar guess"))
                seen_rows.append(
                    f'<div class="apx-row" style="padding:5px 0">{status}<span class="apx-row-main">'
                    f'<b>{escape(SHORT.get(x["festival"], x["festival"]))}</b> · {escape(x["group"])}'
                    f'<span class="apx-row-sub">calendar {_effect(x["calendar"])} → '
                    + (f'your sales show <b>{_effect(x["learned"])}</b> ({x["days_seen"]} days seen)'
                       if x["days_seen"] >= 7 else f'counted as {_effect(x["learned"])} until seen')
                    + "</span></span></div>")
            html("".join(seen_rows))
            html('<div class="apx-label" style="margin-top:12px">Weekday pattern (whole shop)</div>')
            show_chart(fch.weekday_chart(o["shop_dow"]), key="fc_dow")

    with st.container(border=True):
        _title("Can you trust it? Tested on the past",
               f"The model was fitted only on sales before {clock.fmt_day(bt['cutoff'])}, then asked to forecast the 4 weeks after "
               "— which it never saw — and compared with what really sold, and with the usual shop method "
               "(average of the last 30 days).")
        a, b = st.columns([3, 2], gap="large")
        with a:
            show_chart(fch.backtest_weeks_chart(bt), key="fc_bt")
        with b:
            def verdict(x):
                return f"{'over' if x > 0 else 'under'} by {abs(x):.0f}%"
            html(f'<div style="font-size:15px;line-height:1.8">'
                 f'4 weeks from {clock.fmt_day(bt["cutoff"])}: sold <b>{rupees(bt["total_actual"])}</b><br>'
                 f'AI forecast: {rupees(bt["total_predicted"])} — <b>{verdict(bt["total_error_pct"])}</b><br>'
                 f'Simple average: {rupees(bt["total_simple"])} — <b>{verdict(bt["total_error_simple_pct"])}</b></div>')
            html(f'<div class="apx-muted" style="margin-top:10px;line-height:1.6">Repeated over {rb14["windows"]} past '
                 f'2-week periods, <b>per product group</b> (what you order by — interior, exterior, primer…): '
                 f'AI off by <b>{rb14["ai_group_err_pct"]:.0f}%</b>, simple average <b>{rb14["simple_group_err_pct"]:.0f}%</b>; '
                 f'the AI was closer in <b>{rb14["ai_group_wins"]} of {rb14["windows"]}</b>. '
                 f'On the shop\'s total cans: {rb14["ai_err_pct"]:.1f}% vs {rb14["simple_err_pct"]:.1f}% '
                 f'(closer in {rb14["ai_wins"]} of {rb14["windows"]}) — a flat average can get the total right while '
                 f'ordering the wrong mix, which is why the group figure matters. 4-week periods, per group: '
                 f'{rb28["ai_group_err_pct"]:.0f}% vs {rb28["simple_group_err_pct"]:.0f}%.</div>')
        with st.expander("Each 2-week test"):
            show_chart(fch.rolling_errors_chart(rb14["periods"]), key="fc_roll")
            st.caption("Off by, per product group: how many cans were forecast in the wrong group or missed, as a share of what sold.")

    with st.container(border=True):
        _title("One product")
        prods = o["products"]
        opts = {int(r["product_id"]): f"{r['label']} · {r['brand']}" for r in prods.to_dict("records")}
        pid = st.selectbox("Product", list(opts), format_func=opts.get, key="fc_product", label_visibility="collapsed")
        h = data.forecast_product_history(day, int(pid))
        f = data.forecast_product(day, int(pid))
        a, b = st.columns([3, 2], gap="large")
        with a:
            show_chart(fch.product_chart(h, f, max(0.0, 1 - p80), 1 + p80,
                                         [e for e in o["events"] if e["group"] == prods.set_index("product_id").loc[pid, "category"]]),
                       key="fc_prod")
        with b:
            ex = o["model"].explain(int(pid))
            r = prods.set_index("product_id").loc[pid]
            fc = data.forecast(day)
            row = fc[fc["product_id"] == pid]
            lines = [f"<b>{r['units']:.0f} cans</b> expected in 30 days ({rupees(r['revenue'])})",
                     f"{ex['share_pct']}% of {escape(ex['group'].lower())} sales lately",
                     f"Sells most on <b>{ex['best_day']}</b> ({(ex['best_day_factor'] - 1) * 100:+.0f}%), least on "
                     f"{ex['worst_day']} ({(ex['worst_day_factor'] - 1) * 100:+.0f}%)"]
            for e in ex["events"]:
                lines.append(f"{escape(SHORT.get(e['name'], e['name']))}: {_effect(e['effect'])} until {clock.fmt_day(e['until'])}")
            if ex["trend_30d_pct"]:
                lines.append(f"Recent trend: {ex['trend_30d_pct']:+d}% over a month")
            if not row.empty:
                x = row.iloc[0]
                lines.append(f"<b>{int(x['current_stock'])} in stock</b> · lasts about {x['days_to_stockout']:.0f} days · "
                             f"{escape(x['suggested_order_date'])}"
                             + (f" ({int(x['suggested_reorder_qty'])} cans)" if x["suggested_reorder_qty"] else ""))
            html('<div style="line-height:1.9;font-size:14px">' + "<br>".join(lines) + "</div>")

    with st.expander("How the AI forecast works"):
        html('<div class="apx-muted" style="line-height:1.8">'
             "1. Sales are grouped by product type (interior emulsion, exterior, primer, waterproofing, enamel).<br>"
             "2. For each group a regression is fitted on six months of daily sales. It learns the <b>weekday pattern</b>, "
             "how much each <b>festival or season</b> in your calendar really changes sales, and the <b>recent trend</b>.<br>"
             "3. The calendar's numbers are treated as guesses: once a festival has run for a while, your own sales decide its "
             "effect. A festival not seen yet counts at half the calendar's guess.<br>"
             "4. Each group's forecast is shared out across its products by their recent share of sales, and priced.<br>"
             "5. The likely range comes from how far the forecast has been off in past tests.<br>"
             "6. Stock & Reorder uses the same forecast to decide how much to order and by when.</div>")
