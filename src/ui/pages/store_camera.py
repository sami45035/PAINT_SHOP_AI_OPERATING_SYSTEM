"""Store Camera — who came in, and is the shop safe?"""
from html import escape

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src import clock, database as db
from src.camera_stream import SOURCE_LABELS, StoreVisionEngine
from src.config import ESP32_DEFAULT_STREAM_URL, STORE_CLOSE_HOUR, STORE_OPEN_HOUR
from src.ui import data
from src.ui.theme import banner, chip, html, kpi_card, page_header, show_image

SS = st.session_state
EVENT_WORDS = {
    "AFTER_HOURS_MOTION": "Movement after hours",
    "RESTRICTED_ZONE_ENTRY": "Someone in the staff-only zone",
    "CAMERA_DISCONNECTED": "Camera went offline",
}
STATUS_CHIP = {"NEW": ("act_now", "New"), "SEEN": ("neutral", "Seen"), "VERIFIED": ("neutral", "Seen"),
               "RESOLVED": ("healthy", "Resolved"), "FALSE_ALARM": ("healthy", "False alarm")}


def _engine(source: str, url: str) -> StoreVisionEngine:
    key = (source, url if source == "esp32" else "")
    eng = SS.get("cam_engine")
    if eng is None or SS.get("cam_engine_key") != key:
        if eng is not None:
            eng.close_stream()
        eng = StoreVisionEngine(source_type=source, esp32_url=url)
        eng.open_stream()
        SS["cam_engine"], SS["cam_engine_key"] = eng, key
    return eng


def _hour(h: int) -> str:
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


def _feed(eng: StoreVisionEngine) -> None:
    frame, metrics = eng.process_frame()
    if frame is None:
        html('<div class="apx-card" style="background:#26313C;color:#C9CED6;height:300px;display:flex;'
             'align-items:center;justify-content:center;text-align:center">No video</div>')
        return
    show_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    if metrics.get("security_alert"):
        a = metrics["security_alert"]
        saved = "" if a.get("saved") else " (demo — not saved)"
        banner("act_now", "Alert", f"{escape(EVENT_WORDS.get(a['type'], a['type']))}: {escape(a['details'])}{saved}")
    SS["cam_last_metrics"] = metrics


@st.fragment(run_every=2)
def _live_feed() -> None:
    """Re-runs only this part of the page every 2 seconds."""
    eng = SS.get("cam_engine")
    if eng is not None:
        _feed(eng)


def render() -> None:
    page_header("Store Camera", "Counts people at the door and guards the shop after closing.")

    top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
    with top_r:
        source = st.radio("Camera", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get, horizontal=True,
                          index=list(SOURCE_LABELS).index(SS.get("cam_source", "demo")), key="cam_source_radio",
                          label_visibility="collapsed")
        SS["cam_source"] = source
    url = SS.get("esp32_url", ESP32_DEFAULT_STREAM_URL)
    eng = _engine(source, url)

    if source == "demo":
        banner("info", "Demo", escape(eng.status_message))
    elif eng.connected:
        banner("healthy", "Live", escape(eng.status_message))
    else:
        banner("act_now", "Not connected", escape(eng.status_message))
        if st.button("Try again", key="cam_retry"):
            eng.open_stream()
            st.rerun()

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            live = st.toggle("Live view (updates every 2 seconds)", value=True, key="cam_live",
                             disabled=not eng.connected)
            if live and eng.connected:
                _live_feed()
            else:
                _feed(eng)
                if eng.connected and st.button("Refresh frame", key="cam_refresh"):
                    st.rerun()

            if source == "webcam" and not eng.connected:
                snap = st.camera_input("Or take a snapshot with this device's camera", key="cam_browser")
                if snap is not None:
                    img = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    frame, m = eng.process_frame(img)
                    show_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                               caption=f"{m['current_people']} person(s) found in this snapshot")

            with st.expander("Adjust door line and staff-only zone"):
                eng.door_line_y = st.slider("Door line height", 0.2, 0.95, float(eng.door_line_y), 0.01,
                                            help="People crossing this line downward are counted as coming in.")
                z = eng.zone_box
                x1, x2 = st.slider("Staff-only zone — left/right", 0.0, 1.0, (float(z[0]), float(z[2])), 0.01)
                y1, y2 = st.slider("Staff-only zone — top/bottom", 0.0, 1.0, (float(z[1]), float(z[3])), 0.01)
                eng.zone_box = [x1, y1, x2, y2]
            if source == "esp32":
                st.caption(f"Stream address: {url} — change it in Settings (sidebar).")

    with right:
        if source == "demo":
            m = SS.get("cam_last_metrics", {})
            counts = {"in": m.get("entered", 0), "out": m.get("exited", 0)}
            counts["inside"] = max(0, counts["in"] - counts["out"])
            note = "demo counts since this page opened"
        else:
            counts = db.get_door_counts_today()
            note = "today, from the door line"
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Came in", str(counts["in"]), "")
        with c2:
            kpi_card("Left", str(counts["out"]), "")
        with c3:
            kpi_card("Inside now", str(counts["inside"]), "")
        st.caption(note.capitalize())

        with st.container(border=True):
            armed = not clock.is_store_open()
            hdr = chip("act_now", "Armed now") if armed else chip("info", f"Arms at {_hour(STORE_CLOSE_HOUR)}")
            html(f'<div style="display:flex;justify-content:space-between;align-items:center">'
                 f'<span class="apx-step-t" style="font-size:16px">After-hours guard</span>{hdr}</div>'
                 f'<div class="apx-muted" style="margin-top:6px">Any movement between {_hour(STORE_CLOSE_HOUR)} and '
                 f'{_hour(STORE_OPEN_HOUR)} (India time) raises one alert per minute, with a photo.</div>')

        with st.container(border=True):
            html('<span class="apx-step-t" style="font-size:16px">Security log</span>')
            events = db.get_recent_security_events(6)
            if events.empty:
                st.caption("No security events yet.")
            for e in events.itertuples(index=False):
                level, word = STATUS_CHIP.get(e.status, ("neutral", e.status.title()))
                ts = e.timestamp
                try:
                    from datetime import datetime

                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    ts = f"{clock.fmt_day(dt.date())}, {clock.fmt_time(dt)}"
                except ValueError:
                    pass
                html(f'<div class="apx-row" style="align-items:flex-start">{chip(level, word)}<span class="apx-row-main">'
                     f'<b style="font-weight:600">{escape(EVENT_WORDS.get(e.event_type, e.event_type.title()))}</b> · {escape(e.zone)}'
                     f'<span class="apx-row-sub">{escape(ts)}</span></span></div>')
                if e.status == "NEW":
                    b1, b2 = st.columns(2)
                    if b1.button("Seen", key=f"ev_seen_{e.id}", use_container_width=True):
                        db.update_security_event_status(e.id, "SEEN")
                        st.rerun()
                    if b2.button("False alarm", key=f"ev_false_{e.id}", use_container_width=True):
                        db.update_security_event_status(e.id, "FALSE_ALARM")
                        st.rerun()
