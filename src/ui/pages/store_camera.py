"""
Store Camera — the camera as a data source for the shop.

Live            the picture with people, door line, counter, shelf counts, and what just happened
Visitors & buyers   footfall, how many actually bought, where buyers are lost
Shelves & fast movers  cans on each shelf, refill now, what to pre-order, what moves fast,
                    what people pick up but don't buy
Theft & safety  walk-outs with unpaid cans, shelf vs till, after-hours, staff-only area
Set up          place the door line, counter, staff-only area and shelf slots; calibrate shelves
"""
import copy
from datetime import datetime
from html import escape

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src import camera_charts as cc
from src import clock, database as db
from src.camera_stream import COUNT_MODES, SOURCE_LABELS, StoreVisionEngine
from src.config import ESP32_DEFAULT_STREAM_URL, STORE_CLOSE_HOUR, STORE_OPEN_HOUR
from src.formatting import rupees
from src.shop_vision import draw_layout, save_layout
from src.ui import data, nav
from src.ui.components import browser_cam, decode_image
from src.ui.theme import banner, chip, html, kpi_card, page_header, show_chart, show_image

SS = st.session_state
EVENT_WORDS = {
    "AFTER_HOURS_MOTION": "Movement after hours",
    "RESTRICTED_ZONE_ENTRY": "Customer in the staff-only area",
    "CAMERA_DISCONNECTED": "Camera went offline",
    "UNPAID_EXIT": "Left with an unpaid can",
}
STATUS_CHIP = {"NEW": ("act_now", "New"), "SEEN": ("neutral", "Checked"), "VERIFIED": ("neutral", "Checked"),
               "RESOLVED": ("healthy", "Resolved"), "FALSE_ALARM": ("healthy", "False alarm")}
DATA_SOURCES = {"camera": "This camera", "demo": "Demo shop", "sample": "Sample history (30 days)"}
EVENT_CHIP = {"staff": ("neutral", "Staff"), "door": ("info", "Door"), "pick": ("soon", "Picked"), "put_back": ("neutral", "Put back"),
              "restock": ("healthy", "Refill"), "counter": ("info", "Counter"), "visit": ("neutral", "Visit"),
              "alert": ("act_now", "Alert")}


# ----------------------------------------------------------------------------- engine
@st.cache_resource(show_spinner=False)
def _demo_engine() -> StoreVisionEngine:
    """One demo shop for everyone using the app (one background reader, not one per browser tab)."""
    eng = StoreVisionEngine(source_type="demo")
    eng.open_stream()
    return eng


def _engine(source: str, url: str) -> StoreVisionEngine:
    key = (source, url if source == "esp32" else "")
    eng = SS.get("cam_engine")
    if eng is None or SS.get("cam_engine_key") != key:
        if eng is not None and eng.source_type != "demo":
            eng.close_stream()                # the shared demo shop keeps running for others
        if source == "demo":
            eng = _demo_engine()
        else:
            eng = StoreVisionEngine(source_type=source, esp32_url=url)
            eng.open_stream()
        SS["cam_engine"], SS["cam_engine_key"] = eng, key
    if eng.connected and not eng.running:
        eng.start()                           # (re)starts the background reader; it stops itself when nobody watches
    return eng


def _hour(h: int) -> str:
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


def _title(text: str, sub: str = "") -> None:
    html(f'<div class="apx-step-t" style="font-size:17px">{escape(text)}</div>'
         + (f'<div class="apx-muted" style="margin:2px 0 8px">{sub}</div>' if sub else ""))


def _fill_bar(cans: int, cap: int) -> str:
    pct = 0 if cap <= 0 else max(0, min(100, round(cans / cap * 100)))
    colour = "#B42318" if pct <= 30 else ("#B54708" if pct <= 55 else "#067647")
    return (f'<div style="height:8px;border-radius:99px;background:#EEF0F3;overflow:hidden;margin:6px 0 4px">'
            f'<div style="width:{pct}%;height:100%;background:{colour}"></div></div>')


# ----------------------------------------------------------------------------- live
def _right_now(eng: StoreVisionEngine, metrics: dict) -> None:
    counts = db.get_door_counts_today(source=eng.data_source)
    c1, c2, c3 = st.columns(3)
    if eng.count_mode == "presence":
        with c1:
            kpi_card("Came in today", str(counts["in"]), "people seen")
        with c2:
            staff_now = int(metrics.get("staff_in_view", 0))
            kpi_card("In view now", str(max(0, metrics.get("current_people", 0) - staff_now)),
                     f"customers · {staff_now} staff" if staff_now else "at the counter")
        with c3:
            kpi_card("Left today", str(counts["out"]), "visits finished")
        st.caption("Counting **people in view**: someone who stays in view for about 1.5 s counts as a visitor; "
                   "out of view for about 4 s counts as left. Change it in **Set up**.")
    if eng.uniform is not None:
        st.caption(f"Staff in uniform aren't counted — {metrics.get('staff_seen', 0)} recognised since the camera started.")
    else:
        with c1:
            kpi_card("Came in today", str(counts["in"]), "door line")
        with c2:
            kpi_card("In the shop", str(metrics.get("inside_now", counts["inside"])), "customers")
        with c3:
            kpi_card("At the counter", str(metrics.get("at_counter", 0)), "right now")
    shelves = metrics.get("shelves") or {}
    slots = eng.layout.get("shelves", [])
    if slots:
        rows = []
        for s in slots:
            n, cap = shelves.get(s["id"]), int(s.get("capacity", 4))
            label = escape(s.get("label") or s["name"])
            if n is None:
                rows.append(f'<div class="apx-row" style="padding:4px 0"><span class="apx-row-main"><b>{escape(s["id"])}</b> '
                            f'{label}<span class="apx-row-sub">reading…</span></span></div>')
                continue
            rows.append(f'<div style="padding:4px 0"><div style="display:flex;justify-content:space-between;font-size:13px">'
                        f'<span><b>{escape(s["id"])}</b> {label}</span><span style="font-variant-numeric:tabular-nums">'
                        f'<b>{n}</b>/{cap}</span></div>{_fill_bar(n, cap)}</div>')
        html('<div class="apx-label" style="margin-top:12px">Shelves now</div>' + "".join(rows))
    events = metrics.get("events") or []
    html('<div class="apx-label" style="margin-top:12px">Just happened</div>')
    if not events:
        st.caption("Nothing yet — events appear as people come in, pick cans and pay.")
    for e in events[:7]:
        level, word = EVENT_CHIP.get(e["kind"], ("neutral", e["kind"]))
        html(f'<div class="apx-row" style="align-items:flex-start;padding:4px 0">{chip(level, word)}'
             f'<span class="apx-row-main" style="font-size:13px">{escape(e["text"])}'
             f'<span class="apx-row-sub">{escape(e["time"])}</span></span></div>')


def _show_frame(frame) -> None:
    if frame is None:
        html('<div class="apx-card" style="background:#26313C;color:#C9CED6;height:320px;display:flex;'
             'align-items:center;justify-content:center;text-align:center">Starting camera…</div>')
        return
    show_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def _alert_banner(metrics: dict) -> None:
    alert = (metrics or {}).get("security_alert")
    if alert:
        tag = " (demo)" if SS.get("cam_source") == "demo" else ""
        banner("act_now", "Alert", f"{escape(EVENT_WORDS.get(alert['type'], alert['type']))}{tag}: "
                                   f"{escape(alert['details'].split(' Photo:')[0])}")


@st.fragment(run_every=2)
def _live_server(eng: StoreVisionEngine) -> None:
    """Camera read on this computer (webcam / ESP32 / demo) — refreshed every 2 seconds."""
    frame, metrics = eng.snapshot()
    if frame is None and not eng.running:
        frame, metrics = eng.process_frame()
    metrics = metrics or {}
    if metrics.get("security_alert"):
        SS["cam_last_alert"] = metrics["security_alert"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        _alert_banner({"security_alert": SS.get("cam_last_alert")})
        _show_frame(frame)
    with right:
        _right_now(eng, metrics)


def _rerun_fragment() -> None:
    """Rerun just the camera part; during a full-page run Streamlit refuses that, so rerun the page."""
    try:
        st.rerun(scope="fragment")
    except Exception as exc:  # noqa: BLE001
        from src.safety import is_control_flow

        if is_control_flow(exc):
            raise
        st.rerun()


@st.fragment
def _live_browser(eng: StoreVisionEngine) -> None:
    """The camera of the device this page is open on, streamed by the browser (works on Streamlit Cloud)."""
    left, right = st.columns([3, 2], gap="large")
    with left:
        _alert_banner({"security_alert": SS.get("cam_last_alert")})
        door = eng.layout.get("door", {})
        val = browser_cam(key="cam_browser_live", ack=SS.get("bc_ack", 0), boxes=SS.get("bc_boxes", []),
                          line=eng.door_line_y, zone=eng.zone_box, mode=eng.count_mode,
                          door_x=(door.get("x1", 0.0), door.get("x2", 1.0)))
        if val and val.get("seq") and val["seq"] != SS.get("bc_ack"):
            img = decode_image(val.get("frame", ""))
            if img is not None:
                _, m = eng.process_frame(img)
                SS["bc_metrics"], SS["bc_boxes"] = m, m.get("boxes", [])
                if m.get("security_alert"):
                    SS["cam_last_alert"] = m["security_alert"]
            SS["bc_ack"] = val["seq"]
            _rerun_fragment()              # hand the boxes back and ask for the next frame
    with right:
        _right_now(eng, SS.get("bc_metrics") or {})


def _live_tab(eng: StoreVisionEngine, source: str) -> None:
    browser_mode = source == "webcam" and not eng.connected
    if source == "demo":
        banner("info", "Demo shop", "A drawn paint shop on a 64-second loop, read by the same code as a real camera: "
               "a buyer, a browser who puts a can back, a walk-out, an enquiry, and staff refilling the shelves. "
               "Its numbers are saved as <b>Demo shop</b> data, never mixed with real ones.")
    elif browser_mode:
        banner("info", "Browser camera", "Using the camera of the device this page is open on (laptop or phone). "
               "Press <b>Start camera</b> and allow access. Counts and alerts are saved like a shop camera's.")
    elif eng.connected:
        banner("healthy", "Live", escape(eng.status_message))
    else:
        banner("act_now", "Not connected", escape(eng.status_message))
        if st.button("Try again", key="cam_retry"):
            if eng.open_stream():
                eng.start()
            st.rerun()
    if browser_mode:
        _live_browser(eng)
    elif eng.connected:
        _live_server(eng)
    if source == "esp32":
        st.caption(f"Stream address: {SS.get('esp32_url', ESP32_DEFAULT_STREAM_URL)} — change it in Settings (sidebar).")


# ----------------------------------------------------------------------------- data source for the numbers
def _data_source(cam_source: str) -> str:
    have_today = set(db.camera_sources_with_data())
    options = ["sample"]
    if "camera" in have_today or cam_source in ("webcam", "esp32"):
        options.insert(0, "camera")
    if "demo" in have_today or cam_source == "demo":
        options.insert(0, "demo")
    default = "demo" if cam_source == "demo" else ("camera" if "camera" in have_today else "sample")
    if SS.get("cam_data_src") not in options or SS.get("cam_data_src_for") != cam_source:
        SS["cam_data_src"] = default           # follow the camera choice until the user picks otherwise
        SS["cam_data_src_for"] = cam_source
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        src = st.radio("Numbers from", options, format_func=DATA_SOURCES.get, horizontal=True, key="cam_data_src")
    with c2:
        if st.button("Refresh numbers", key="cam_refresh_numbers", use_container_width=True):
            data.clear_all()
            st.rerun()
    if src == "sample":
        st.caption("Sample history: 30 days of camera data built to match the sample sales, so every screen has "
                   "something to show. Switch to *This camera* once a camera has been running.")
    elif src == "demo":
        st.caption("Demo shop: what the drawn demo has recorded since it started. There's no till in the demo, "
                   "so 'bought' is shown as 'reached the counter'.")
    return src


# ----------------------------------------------------------------------------- visitors & buyers
def _buyers_tab(src: str, day: str) -> None:
    f = data.cam_footfall(src, day)
    b = data.cam_buyers(src, day)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        vs = f["vs_last_week_pct"]
        sub = (f'<span class="{"apx-up" if vs >= 0 else "apx-down"}">{"▲" if vs >= 0 else "▼"} {abs(vs)}%</span> vs last '
               f'{clock.now():%A}' if vs is not None else "today")
        kpi_card("Came in", str(f["came_in"]), sub)
    with c2:
        if b.get("classified"):
            kpi_card("Bought", f"{b['bought_visit_pct']:.0f}%",
                     f"{b['bought_visits']} of {b['finished_visits']} visitors matched to a bill · {rupees(b['bought_value'])}")
        elif b["real_till"]:
            pct = b["bought_pct"]
            if pct is not None and pct > 100:
                kpi_card("Bought", "—", f"{b['bills']} bills but only {b['visitors']} visitors counted — "
                                        "the camera isn't seeing everyone yet")
            else:
                kpi_card("Bought", f"{pct:.0f}%" if pct is not None else "—",
                         f"{b['bills'] or 0} bills ÷ {b['visitors']} visitors"
                         + (f" since {b['since'][11:16]}" if src == "camera" else ""))
        else:
            kpi_card("Reached the counter", str(b["at_counter"]), "no till in the demo")
    with c3:
        if b.get("classified"):
            top = max(((k, n) for k, n in b["reasons"].items() if k != "unknown"), key=lambda kv: kv[1], default=None)
            kpi_card("Didn't buy", str(b["didnt_buy"]), f"of {b['finished_visits']} visits · top reason: "
                     + (REASON_WORDS.get(top[0], top[0]).lower() if top else "not recorded"),
                     attention=b["didnt_buy"] > b["bought_visits"])
        else:
            kpi_card("Left without the counter", str(b["left_without_counter"]),
                     f"of {b['finished_visits']} visits finished", attention=b["left_without_counter"] > b["at_counter"])
    with c4:
        kpi_card("Average stay", f"{f['avg_stay_min']:.0f} min" if f["avg_stay_min"] else "—", "per visit")
    with c5:
        pk = f["peak"]
        kpi_card("Busiest hour", pk["label"] if pk else "—", f"{pk['visitors']} came in" if pk else "")

    st.write("")
    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            _title("Where visitors go", "Today, from the door to the till")
            show_chart(cc.funnel_chart(b), key="cam_funnel")
    with right:
        with st.container(border=True):
            _title("Came in vs bought, by hour")
            show_chart(cc.hourly_buyers_chart(b["hourly"], b["real_till"]), key="cam_hourly")

    if b.get("classified"):
        _who_bought(b)

    tips = []
    for w in b["weak_hours"]:
        val = f" — about {rupees(round(w['missed_bills'] * b['avg_bill']))} in sales" if b["avg_bill"] else ""
        tips.append(("soon", f"{w['label']}: {w['visitors']} came in, {w['bills']} bought",
                     f"At today's usual rate that's {w['missed_bills']} more bill{'s' if w['missed_bills'] != 1 else ''}{val}. "
                     "Was the counter busy, or a popular shade out? Put a second person on the floor at this hour."))
    if b["visitors"] and b["at_shelves"] < b["visitors"] * 0.4:
        tips.append(("look_into", "Most visitors don't reach the shelves",
                     f"Only {b['at_shelves']} of {b['visitors']} stood at a watched shelf. Move the best sellers nearer the door."))
    if f["peak"]:
        tips.append(("info", f"Busiest at {f['peak']['label']}",
                     "Keep the counter staffed and the fast movers stocked before this hour."))
    if tips:
        cols = st.columns(min(3, len(tips)))
        for col, (lvl, t, d) in zip(cols, tips[:3]):
            with col:
                with st.container(border=True):
                    html(f'{chip(lvl, "Insight" if lvl == "info" else None)}<div style="font-weight:600;margin:8px 0 4px">'
                         f'{escape(t)}</div><div class="apx-muted" style="line-height:1.5">{escape(d)}</div>')
    if b["real_till"]:
        with st.container(border=True):
            _title("Visitors who bought · last 14 days")
            show_chart(cc.daily_bought_chart(data.cam_daily(src, day)), key="cam_daily")


REASON_WORDS = {"price": "Price too high", "not_in_stock": "Shade / pack not in stock", "just_asking": "Just asking",
                "come_back": "Will come back", "other": "Other", "unknown": "No reason recorded"}


def _who_bought(b: dict) -> None:
    """Each visitor: bought (a bill matched by time at the counter) or didn't — and why."""
    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            _title("Who bought", "Each visit is matched to the bill saved while that customer was at the counter")
            rows = [("healthy", "Bought", b["bought_visits"], rupees(b["bought_value"])),
                    ("act_now", "Didn't buy", b["didnt_buy"], "")]
            html("".join(f'<div class="apx-row">{chip(l, t)}<span class="apx-row-main"><b>{n}</b> visitor{"s" if n != 1 else ""}'
                         f'</span><span class="apx-row-right">{v}</span></div>' for l, t, n, v in rows))
            if b["unmatched_bills"]:
                st.caption(f"{b['unmatched_bills']} bill(s) had nobody at the counter on camera (camera off, or billed later) "
                           "— counted in sales, not linked to a visit.")
            st.caption("People who came in together share one bill.")
    with right:
        with st.container(border=True):
            _title("Why they didn't buy", "From the one-tap 'Didn't buy' on Scan & Bill")
            show_chart(cc.reasons_chart(b["reasons"], REASON_WORDS), key="cam_reasons")
            nav.link("bill", "Open Scan & Bill →")


# ----------------------------------------------------------------------------- shelves & fast movers
def _shelves_tab(src: str, day: str) -> None:
    sh = data.cam_shelves(src, day)
    po = data.cam_preorder(src, day)
    fm = data.cam_fast_movers(src, day)

    with st.container(border=True):
        _title("Shelves right now", "Cans counted on each watched shelf; the store room is what the stock book has beyond that")
        if sh.empty:
            st.info("No shelves are watched yet. In **Set up**, draw a box around each shelf slot and pick its product.")
        else:
            cols = st.columns(min(6, len(sh)))
            for i, r in enumerate(sh.to_dict("records")):
                with cols[i % len(cols)]:
                    cap = int(r["capacity"])
                    low = r["cans"] <= max(1, round(cap * 0.3))
                    html(f'<div class="apx-card{" attention" if low else ""}" style="padding:12px">'
                         f'<div style="display:flex;gap:8px;align-items:center"><span class="apx-sw" style="background:'
                         f'{escape(r.get("hex_code") or "#ccc")}"></span><span style="font-weight:600;font-size:13px">'
                         f'{escape(r["slot_id"])} · {escape(r["label"])}</span></div>'
                         f'<div class="apx-kpi" style="font-size:24px">{r["cans"]}<span class="apx-muted" style="font-size:14px">'
                         f' / {cap}</span></div>{_fill_bar(r["cans"], cap)}'
                         f'<div class="apx-muted" style="font-size:12px">{r["store_room"]} in store room'
                         + (f' · {r["on_order"]} on order' if r["on_order"] else "") + "</div></div>")

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            _title("Refill now", "Low shelves with cans waiting in the store room")
            if not po["refill"]:
                html(f'{chip("healthy", "All good")} <span class="apx-muted">Every watched shelf is stocked.</span>')
            for i, r in enumerate(po["refill"]):
                c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
                with c1:
                    html(f'<div class="apx-row" style="align-items:flex-start">{chip(r["level"])}<span class="apx-row-main">'
                         f'{escape(r["message"])}</span></div>')
                with c2:
                    if r["move"] and st.button("Refilled", key=f"cam_refill_{i}", use_container_width=True,
                                               help=f"Records {r['move']} cans moved to {r['slot_name']}"):
                        db.log_shelf_event(src, r["slot_id"], int(r["product_id"]) if r["product_id"] else None,
                                           "restock", int(r["move"]))
                        data.clear_all()
                        st.rerun()
    with right:
        with st.container(border=True):
            _title("Pre-order", "From the sales forecast, supplier lead times and the camera's pick-up trend")
            if not po["order"]:
                html(f'{chip("healthy", "Nothing due")} <span class="apx-muted">No product runs out before its supplier can deliver.</span>')
            for i, o in enumerate(po["order"][:6]):
                c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
                with c1:
                    cam = " 📷" if o["watched"] else ""
                    html(f'<div class="apx-row" style="align-items:flex-start"><span class="apx-sw" style="background:'
                         f'{escape(o["hex_code"])}"></span><span class="apx-row-main"><b>{escape(o["label"])}</b>{cam} · '
                         f'{escape(o["when"])} · <b>{o["qty"]} cans</b> ({rupees(o["cost"])})'
                         f'<span class="apx-row-sub">{escape(o["message"])}</span></span></div>')
                with c2:
                    if st.button("Ordered", key=f"cam_order_{i}", use_container_width=True,
                                 help="Adds these cans to 'on order'"):
                        db.mark_ordered(o["product_id"], o["qty"])
                        data.clear_all()
                        st.toast(f"{o['qty']} × {o['label']} marked as ordered.")
                        st.rerun()
            if len(po["order"]) > 6:
                st.caption(f"{len(po['order']) - 6} more on the Stock & Reorder screen. 📷 = shelf watched by the camera.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            _title("Fast movers · last 7 days", "From the till; picked up / put back from the camera for watched shelves")
            top = fm["top"]
            if top.empty:
                st.caption("No sales in the last 7 days.")
            rows = []
            for r in top.to_dict("records"):
                tr = r.get("trend_pct")
                trend = "" if tr is None or tr != tr else (
                    f'<span class="{"apx-up" if tr >= 0 else "apx-down"}">{"▲" if tr >= 0 else "▼"} {abs(tr):.0f}%</span>')
                picked = r.get("picked")
                cam = "" if picked is None or picked != picked else (
                    f' · 📷 picked up {int(picked)}×, put back {int(r["put_back_pct"])}%')
                rows.append(f'<div class="apx-row"><span class="apx-sw" style="background:{escape(r["hex_code"])}"></span>'
                            f'<span class="apx-row-main"><b>{escape(r["label"])}</b> <span class="apx-muted">· {escape(r["brand"])}</span>'
                            f'<span class="apx-row-sub">{r["per_day"]:.1f} a day{cam}</span></span>'
                            f'<span class="apx-row-right">{int(r["units"])} cans {trend}</span></div>')
            html("".join(rows))
    with right:
        with st.container(border=True):
            _title("Picked up, put back, sold", "Watched shelves, last 7 days")
            show_chart(cc.picks_vs_sold_chart(fm["watched"]), key="cam_picks")
        for h in fm["hesitant"][:2]:
            with st.container(border=True):
                html(f'{chip("look_into", "Picked up, not bought")}<div style="font-weight:600;margin:8px 0 4px">'
                     f'{escape(h["label"])}</div><div class="apx-muted" style="line-height:1.5">{escape(h["message"])}</div>')
        for r in fm["rising"][:2]:
            with st.container(border=True):
                html(f'{chip("info", "Demand building")}<div style="font-weight:600;margin:8px 0 4px">{escape(r["label"])}</div>'
                     f'<div class="apx-muted" style="line-height:1.5">{escape(r["message"])}</div>')


# ----------------------------------------------------------------------------- theft & safety
def _security_log(src: str) -> None:
    events = data.cam_theft(src, clock.today_str())["events"]
    if events.empty:
        st.caption("No security events in the last 7 days.")
    for e in events.head(8).itertuples(index=False):
        level, word = STATUS_CHIP.get(e.status, ("neutral", e.status.title()))
        try:
            dt = datetime.strptime(e.timestamp, "%Y-%m-%d %H:%M:%S")
            ts = f"{clock.fmt_day(dt.date())}, {clock.fmt_time(dt)}"
        except ValueError:
            ts = e.timestamp
        details = (e.details or "").split(" Photo:")[0]
        html(f'<div class="apx-row" style="align-items:flex-start">{chip(level, word)}<span class="apx-row-main">'
             f'<b style="font-weight:600">{escape(EVENT_WORDS.get(e.event_type, e.event_type.title()))}</b> · {escape(e.zone)}'
             f'<span class="apx-row-sub">{escape(ts)} — {escape(details)}</span></span></div>')
        if e.status == "NEW":
            b1, b2 = st.columns(2)
            if b1.button("Checked", key=f"ev_seen_{e.id}", use_container_width=True):
                db.update_security_event_status(e.id, "SEEN")
                data.clear_all()
                st.rerun()
            if b2.button("False alarm", key=f"ev_false_{e.id}", use_container_width=True):
                db.update_security_event_status(e.id, "FALSE_ALARM")
                data.clear_all()
                st.rerun()


def _theft_tab(src: str, day: str) -> None:
    th = data.cam_theft(src, day)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Walk-outs with a can", str(th["walkouts_today"]), "today", attention=th["walkouts_today"] > 0,
                 value_colour="#B42318" if th["walkouts_today"] else None)
    with c2:
        kpi_card("Shelf vs till gap", rupees(th["unbilled_value"]), "today, cans gone without a bill",
                 attention=th["unbilled_value"] > 0)
    with c3:
        kpi_card("Alerts to check", str(th["open"]), "last 7 days", attention=th["open"] > 0)
    with c4:
        kpi_card("After hours / staff area", f"{th['after_hours']} / {th['staff_zone']}", "last 7 days")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            _title("Shelf vs till · today",
                   "Cans that left the watched shelves (not put back) compared with what the till billed")
            if not th["shelf_vs_till"]:
                html(f'{chip("healthy", "Matches")} <span class="apx-muted">Every can that left a watched shelf today '
                     "has a bill (cans picked in the last 45 minutes are left out — the customer may not have paid yet).</span>")
            for g in th["shelf_vs_till"]:
                missing_chip = chip("soon", str(g["missing"]) + " missing")
                html(f'<div class="apx-row">{missing_chip}<span class="apx-row-main"><b>{escape(g["label"])}</b>'
                     f'<span class="apx-row-sub">{g["taken"]} taken off the shelf · {g["sold"]} billed · about '
                     f'{rupees(g["value"])}</span></span></div>')
            st.caption("A gap can also be a missed bill or a can moved to the store room — check before blaming anyone.")
        with st.container(border=True):
            _title("How the camera spots theft")
            html('<div class="apx-muted" style="line-height:1.7">'
                 "• <b>Walk-out:</b> a can disappears from a shelf while someone stands there, and that person crosses the door "
                 "line going out without having stood at the billing counter → alert with a photo.<br>"
                 "• <b>Shelf vs till:</b> every can taken (and not put back) is compared with the day's bills.<br>"
                 f"• <b>After hours:</b> any movement between {_hour(STORE_CLOSE_HOUR)} and {_hour(STORE_OPEN_HOUR)}.<br>"
                 "• <b>Staff-only area:</b> a customer (someone who came in through the door) standing in it.</div>")
    with right:
        with st.container(border=True):
            _title("Security log", "Last 7 days")
            _security_log(src)


# ----------------------------------------------------------------------------- set up
ZONE_KINDS = ["Shelf slot", "Billing counter", "Staff-only area", "Door line"]


@st.cache_data(ttl=300, show_spinner=False)
def _product_choices():
    df = db.get_all_products()
    df = df[df["category"] != "Waterproofing"]
    return {int(r.id): f"{r.shade_name} {r.pack_size_litres:g} L · {r.brand}" for r in df.itertuples(index=False)}


def _uniform_setup(eng: StoreVisionEngine) -> None:
    """Staff wear the shop uniform -> the camera doesn't count them as customers."""
    with st.container(border=True):
        _title("Staff uniform — don't count staff as customers")
        u = dict(eng.layout.get("uniform") or {})
        on = st.toggle("Staff wear a uniform (shirt / T-shirt / apron in one colour)", value=bool(u.get("on")),
                       key="cam_uni_on")
        colour = u.get("hex") or "#1E78C8"
        if on:
            c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
            with c1:
                colour = st.color_picker("Uniform colour", colour, key=f"cam_uni_hex_{colour}")
            with c2:
                if st.button("Capture from the camera", key="cam_uni_capture", use_container_width=True,
                             help="Ask a staff member in uniform to stand facing the camera, then press this"):
                    got = eng.capture_uniform()
                    if got:
                        colour = got
                        st.toast(f"Uniform colour captured: {got}")
                    else:
                        st.warning("No one found in the picture — start the camera on the Live tab and stand in front of it.")
            hsv = cv2.cvtColor(np.uint8([[[int(colour[5:7], 16), int(colour[3:5], 16), int(colour[1:3], 16)]]]),
                               cv2.COLOR_BGR2HSV)[0, 0]
            if hsv[1] < 90 or hsv[2] < 80:
                st.warning("This colour is dull or dark — customers' clothes (grey, brown, black, beige) can look like it. "
                           "A bright uniform (orange, green, blue, red) is recognised far more reliably.")
            st.caption("Works best with a bright colour customers rarely wear (e.g. the shop's orange or green T-shirt). "
                       "Someone is marked **Staff** after the camera sees the uniform on them for about a second; "
                       "staff aren't counted as visitors, aren't matched to bills, and don't set off the store-room alert.")
            frame = eng.latest_raw
            if frame is not None:
                from src.camera_stream import shirt_region, uniform_share, hex_to_bgr, UNIFORM_MATCH

                h, w = frame.shape[:2]
                bodies = eng.detect_people(frame)
                found = [(b, "body") for b in bodies] + [(f, "face") for f in eng.detect_faces(frame, bodies)]
                checks = [uniform_share(frame, r, hex_to_bgr(colour)) for r in
                          (shirt_region(b, k, w, h) for b, k in found) if r is not None]
                if checks:
                    staff = sum(c >= UNIFORM_MATCH for c in checks)
                    st.caption(f"Right now: {len(checks)} person(s) in view — {staff} wearing the uniform.")
        new = {"on": bool(on), "hex": colour}
        if new != {"on": bool(u.get("on")), "hex": u.get("hex") or "#1E78C8"}:
            lay = copy.deepcopy(eng.layout)
            lay["uniform"] = new
            save_layout(lay)
            eng.set_layout(lay)
            SS["cam_layout_edit"] = copy.deepcopy(lay)
            st.rerun()


def _setup_tab(eng: StoreVisionEngine, source: str) -> None:
    if source == "demo":
        st.info("The demo shop's zones are fixed. Switch the camera to **Webcam** or **ESP32-CAM** to set up your own.")
        frame = eng.latest_raw
        if frame is not None:
            show_image(Image.fromarray(cv2.cvtColor(draw_layout(frame, eng.layout), cv2.COLOR_BGR2RGB)))
        return
    with st.container(border=True):
        _title("How should this camera count people?")
        modes = list(COUNT_MODES)
        mode = st.radio("Counting", modes, index=modes.index(eng.count_mode), format_func=COUNT_MODES.get,
                        key="cam_count_mode", label_visibility="collapsed")
        if mode != eng.count_mode:
            lay = copy.deepcopy(eng.layout)
            lay["count_mode"] = mode
            save_layout(lay)
            eng.set_layout(lay)
            SS["cam_layout_edit"] = copy.deepcopy(lay)
            st.toast("Counting mode saved.")
            st.rerun()
        st.caption("A laptop webcam or a camera above the billing counter sees people's faces and upper bodies — "
                   "use **People in view**. A camera that sees the whole doorway from above — use **Door line**, "
                   "which also tells coming in from going out and follows each customer to the counter.")
    _uniform_setup(eng)
    frame = eng.latest_raw
    if frame is None:
        st.info("Start the camera on the **Live** tab first — placing zones needs a picture of the shop.")
        return
    lay = SS.setdefault("cam_layout_edit", copy.deepcopy(eng.layout))
    pending = SS.setdefault("cam_clicks", [])

    left, right = st.columns([3, 2], gap="large")
    with right:
        with st.container(border=True):
            _title("Place a zone", "Pick what to place, then tap two corners on the picture")
            kind = st.radio("What", ZONE_KINDS, key="cam_zone_kind", label_visibility="collapsed")
            if kind == "Shelf slot":
                prods = _product_choices()
                st.selectbox("Product on this slot", list(prods), format_func=prods.get, key="cam_slot_product")
                st.number_input("Cans that fit side by side", 1, 30, 8, 1, key="cam_slot_capacity")
                st.caption("One slot = one product, cans standing side by side, fronts facing the camera.")
            elif kind == "Door line":
                st.caption("Tap the left and right ends of the doorway, at the height where people's feet cross it.")
            else:
                st.caption("Tap two opposite corners of the floor area where people stand.")
            if pending:
                st.caption(f"{len(pending)} of 2 corners placed.")
                if st.button("Start over", key="cam_clicks_reset"):
                    SS["cam_clicks"] = []
                    st.rerun()
        with st.container(border=True):
            _title("Shelf slots")
            if not lay.get("shelves"):
                st.caption("None yet.")
            for i, s in enumerate(list(lay.get("shelves", []))):
                c1, c2 = st.columns([4, 1], vertical_alignment="center")
                c1.markdown(f"**{s['id']}** · {s.get('label', '')} · {s.get('capacity', 8)} cans")
                if c2.button("✕", key=f"cam_del_slot_{i}", help="Remove this slot"):
                    lay["shelves"].pop(i)
                    st.rerun()
        b1, b2 = st.columns(2)
        if b1.button("Save set-up", type="primary", key="cam_save_layout", use_container_width=True):
            save_layout(lay)
            eng.set_layout(copy.deepcopy(lay))
            st.toast("Set-up saved — the camera uses it now.")
        if b2.button("Undo changes", key="cam_undo_layout", use_container_width=True):
            SS["cam_layout_edit"] = copy.deepcopy(eng.layout)
            SS["cam_clicks"] = []
            st.rerun()
        with st.container(border=True):
            _title("Calibrate shelves", "Do this once the slots are saved, with the shelves as they are now")
            slots = eng.layout.get("shelves", [])
            counts = {}
            for s in slots:
                counts[s["id"]] = st.number_input(f"{s['id']} · cans on it now", 0, int(s.get("capacity", 8)),
                                                  int(s.get("capacity", 8)), 1, key=f"cam_cal_{s['id']}")
            if slots and st.button("Calibrate now", key="cam_calibrate", use_container_width=True):
                done = eng.shelves.calibrate(eng.latest_raw, counts)
                for s in slots:
                    if s["id"] in done:
                        db.set_shelf_status(eng.data_source, s["id"], s["name"], s.get("product_id"), done[s["id"]],
                                            int(s.get("capacity", 8)))
                data.clear_all()
                st.toast(f"Calibrated {len(done)} shelf slot{'s' if len(done) != 1 else ''}.")
    with left:
        shown = draw_layout(frame, lay, pending)
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        disp_w = 720
        pil = Image.fromarray(rgb).resize((disp_w, int(h * disp_w / w)))
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
        except ImportError:  # pragma: no cover
            show_image(pil)
            st.caption("Install streamlit-image-coordinates to place zones by tapping.")
            return
        click = streamlit_image_coordinates(pil, key="cam_setup_click")
        if click and (click.get("x"), click.get("y"), click.get("unix_time")) != SS.get("cam_last_click"):
            SS["cam_last_click"] = (click.get("x"), click.get("y"), click.get("unix_time"))
            pw, ph = click.get("width") or disp_w, click.get("height") or pil.height
            pending.append((round(click["x"] / pw, 4), round(click["y"] / ph, 4)))
            if len(pending) == 2:
                (xa, ya), (xb, yb) = pending
                x1, x2, y1, y2 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
                kind = SS.get("cam_zone_kind", ZONE_KINDS[0])
                if kind == "Door line":
                    lay["door"] = {"y": round((ya + yb) / 2, 4), "x1": x1, "x2": x2}
                elif kind == "Billing counter":
                    lay["counter"] = [x1, y1, x2, y2]
                elif kind == "Staff-only area":
                    lay["staff_zone"] = [x1, y1, x2, y2]
                else:
                    pid = SS.get("cam_slot_product")
                    used = {s["id"] for s in lay.get("shelves", [])}
                    sid = next(f"S{i}" for i in range(1, 99) if f"S{i}" not in used)
                    label = _product_choices().get(pid, "").split(" · ")[0]
                    lay.setdefault("shelves", []).append({
                        "id": sid, "name": f"Shelf {sid}", "box": [x1, y1, x2, y2], "product_id": pid,
                        "capacity": int(SS.get("cam_slot_capacity", 8)), "label": label})
                SS["cam_clicks"] = []
            st.rerun()
        st.caption("Unsaved changes are drawn here; press **Save set-up** to use them.")


# ----------------------------------------------------------------------------- page
def render() -> None:
    page_header("Store Camera", "Footfall, who actually buys, shelf stock, fast movers and theft — from one camera.")
    top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
    with top_r:
        source = st.radio("Camera", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get, horizontal=True,
                          index=list(SOURCE_LABELS).index(SS.get("cam_source", "demo")), key="cam_source_radio",
                          label_visibility="collapsed")
        SS["cam_source"] = source
    with top_l:
        if source == "demo" and st.button("Clear demo numbers", key="cam_clear_demo",
                                          help="Forget everything the demo shop has recorded"):
            db.clear_camera_data("demo")
            data.clear_all()
            st.rerun()
    url = SS.get("esp32_url", ESP32_DEFAULT_STREAM_URL)
    eng = _engine(source, url)
    day = clock.today_str()

    tabs = st.tabs(["Live", "Visitors & buyers", "Shelves & fast movers", "Theft & safety", "Set up"])
    with tabs[0]:
        _safely("Live view", _live_tab, eng, source)
    with tabs[1]:
        src = _data_source(source)
        _safely("Visitors & buyers", _buyers_tab, src, day)
    with tabs[2]:
        st.caption(f"Numbers from: **{DATA_SOURCES[src]}** (change on the Visitors & buyers tab)")
        _safely("Shelves & fast movers", _shelves_tab, src, day)
    with tabs[3]:
        st.caption(f"Numbers from: **{DATA_SOURCES[src]}** (change on the Visitors & buyers tab)")
        _safely("Theft & safety", _theft_tab, src, day)
    with tabs[4]:
        _safely("Set up", _setup_tab, eng, source)


def _safely(name: str, fn, *args) -> None:
    """One tab failing must not take the others (or the live camera) down with it."""
    from src.safety import guard

    guard(name, fn, *args)
