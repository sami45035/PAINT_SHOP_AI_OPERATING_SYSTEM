"""
Paint Shop AI — Apex Colors & Paints Studio
Streamlit entry point: theme, sidebar and navigation for the six screens.

L1 Shop data · L2 Overview · L3 Forecast & reorder · L4 Alerts · L5 Store Camera
L6 Colour Studio · L7 Ask Aura

Run:  streamlit run app.py
"""
import faulthandler
import os
import subprocess
import sys
from html import escape

# If a compiled library (OpenCV, pyarrow, ONNX Runtime…) crashes the process,
# print where it happened to the log instead of dying silently.
faulthandler.enable(file=sys.stderr, all_threads=True)

import pandas as pd  # noqa: E402

try:
    # pandas 3 stores text via pyarrow when it's installed; keep the classic
    # storage — sturdier across pyarrow versions on Streamlit Cloud.
    pd.set_option("future.infer_string", False)
except Exception:
    pass

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Apex Paint AI", page_icon="🎨", layout="wide", initial_sidebar_state="expanded")

from src import clock  # noqa: E402
from src.config import DB_PATH, DEMO_MODE, ESP32_DEFAULT_STREAM_URL, STORE_SHORT_NAME  # noqa: E402
from src.ui import data, nav  # noqa: E402
from src.ui.theme import html, inject_css  # noqa: E402


@st.cache_resource(show_spinner=False)
def _log_versions() -> bool:
    """One line in the server log with the versions in use (helps when something breaks)."""
    import importlib

    found = []
    for name in ("streamlit", "pandas", "numpy", "pyarrow", "cv2", "PIL", "plotly", "onnxruntime"):
        try:
            found.append(f"{name} {importlib.import_module(name).__version__}")
        except Exception as exc:  # noqa: BLE001
            found.append(f"{name} unavailable ({type(exc).__name__})")
    print(f"[apex] Python {sys.version.split()[0]} | " + " | ".join(found), file=sys.stderr, flush=True)
    return True


# Each screen's libraries load only when that screen is opened, so a problem in
# one (e.g. the camera's OpenCV) can't stop the whole app from starting.
def _overview():
    from src.ui.pages import overview
    overview.render()


def _colour_studio():
    from src.ui.pages import colour_studio
    colour_studio.render()


def _stock():
    from src.ui.pages import stock_reorder
    stock_reorder.render()


def _alerts():
    from src.ui.pages import alerts
    alerts.render()


def _camera():
    from src.ui.pages import store_camera
    store_camera.render()


def _aura():
    from src.ui.pages import ask_aura
    ask_aura.render()


@st.cache_resource(show_spinner="Preparing shop data…")
def _prepare_database(day: str) -> int:
    """Create the sample database if missing; keep demo dates current (once per day)."""
    from src import database as db

    if not DB_PATH.exists():
        subprocess.run([sys.executable, str(DB_PATH.parent / "seed_data.py")], check=True)
    db.ensure_app_tables()
    return db.ensure_demo_dates_current()


_prepare_database(clock.today_str())
inject_css()

_log_versions()

pages = {
    "overview": st.Page(_overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    "colour": st.Page(_colour_studio, title="Colour Studio", icon=":material/format_paint:", url_path="colour-studio"),
    "stock": st.Page(_stock, title="Stock & Reorder", icon=":material/inventory_2:", url_path="stock"),
    "alerts": st.Page(_alerts, title="Alerts", icon=":material/notifications:", url_path="alerts"),
    "camera": st.Page(_camera, title="Store Camera", icon=":material/videocam:", url_path="camera"),
    "aura": st.Page(_aura, title="Ask Aura", icon=":material/chat:", url_path="ask-aura"),
}
nav.PAGES.update(pages)
current = st.navigation(list(pages.values()), position="hidden")

with st.sidebar:
    html(
        '<div class="apx-brand"><div class="apx-mark"><span style="height:14px;background:#fff"></span>'
        '<span style="height:20px;background:#D9B48C"></span><span style="height:10px;background:#9CAF88"></span></div>'
        f'<div><div class="apx-brand-name">{escape(STORE_SHORT_NAME)}</div><div class="apx-brand-sub">Paint Shop AI</div></div></div>'
    )
    is_open = clock.is_store_open()
    fg, bg = ("#067647", "#E5F3EB") if is_open else ("#B42318", "#FCEBE9")
    html(f'<div class="apx-pill" style="background:{bg};color:{fg};margin-bottom:12px">'
         f'<span class="apx-dot" style="background:{fg}"></span>{escape(clock.opening_label())}</div>')

    try:
        n_alerts = len(data.open_alerts(clock.today_str()))
    except Exception:
        n_alerts = 0
    for key, page in pages.items():
        label = page.title + (f"  ·  {n_alerts}" if key == "alerts" and n_alerts else "")
        st.page_link(page, label=label, icon=page.icon)

    st.divider()
    cam = {"webcam": "Laptop webcam", "esp32": "ESP32-CAM", "demo": "Demo feed"}[st.session_state.get("cam_source", "demo")]
    gem = bool(st.session_state.get("gemini_key") or os.getenv("GEMINI_API_KEY"))
    aura = "Gemini" if gem and st.session_state.get("aura_mode") == "Gemini" else "Offline mode"
    now = clock.now()
    html(
        f'<div class="apx-foot">Camera · <b>{cam}</b><br>Aura · <b>{aura}</b><br>'
        f'Data as of · <b>{clock.fmt_day(now.date())}, {clock.fmt_time(now)}</b>'
        + ("<br>Sample data · <b>demo mode</b>" if DEMO_MODE else "") + "</div>"
    )
    with st.expander("Settings"):
        st.text_input("Gemini API key", type="password", key="gemini_key",
                      help="Optional. Lets Aura answer free-form questions. Stays in this browser session.")
        st.session_state.setdefault("esp32_url", ESP32_DEFAULT_STREAM_URL)
        st.text_input("ESP32-CAM stream address", key="esp32_url")
        if st.button("Refresh data", use_container_width=True):
            data.clear_all()
            st.rerun()

current.run()
