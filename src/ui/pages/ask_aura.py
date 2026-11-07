"""Ask Aura — ask anything about the shop in plain words."""
import os
from html import escape

import streamlit as st

from src.ai_assistant import GENAI_AVAILABLE, answer, deterministic_assistant_response
from src.ui.theme import html, page_header

SS = st.session_state
PROMPTS = ["How did we do today?", "Which shades are running low?", "What needs attention?",
           "Suggest a calm bedroom colour", "Do we have Royale Blue?"]
MODE_LABEL = {"gemini": "Gemini", "offline": "Offline mode", "offline_fallback": "Offline (Gemini didn't respond)"}


def _api_key() -> str:
    return SS.get("gemini_key") or os.getenv("GEMINI_API_KEY", "")


def _ask(question: str) -> None:
    history = SS.setdefault("aura_history", [])
    use_gemini = SS.get("aura_mode", "Offline") == "Gemini"
    with st.spinner("Aura is checking the shop data…"):
        if use_gemini:
            res = answer(question, history, api_key=_api_key())
        else:
            res = {"text": deterministic_assistant_response(question), "mode": "offline"}
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": res["text"], "mode": res["mode"]})


def render() -> None:
    page_header("Ask Aura", "Your shop assistant. It reads the same stock, sales and alerts you see.")
    has_key = bool(_api_key()) and GENAI_AVAILABLE

    top_l, top_r = st.columns([3, 1], vertical_alignment="bottom")
    with top_r:
        mode = st.radio("Mode", ["Offline", "Gemini"], horizontal=True, key="aura_mode", label_visibility="collapsed")

    chat_col, side = st.columns([3, 1.2], gap="large")
    with chat_col:
        with st.container(border=True, height=520):
            history = SS.get("aura_history", [])
            if not history:
                with st.chat_message("assistant"):
                    st.markdown("Hi! Ask me about stock, reorders, alerts, today's numbers or colours.")
            for m in history:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m["role"] == "assistant":
                        st.caption(f"Based on live shop data · {MODE_LABEL.get(m.get('mode'), '')}")
        picked = st.pills("Try asking", PROMPTS, key="aura_pick", label_visibility="collapsed") \
            if hasattr(st, "pills") else None
        typed = st.chat_input("Ask about stock, sales, colours…")
        question = typed or (picked if picked and picked != SS.get("aura_last_pick") else None)
        if picked:
            SS["aura_last_pick"] = picked
        if question:
            _ask(question)
            st.rerun()

    with side:
        with st.container(border=True):
            html('<span class="apx-step-t" style="font-size:16px">Aura can see</span>')
            for name, state in [("Stock & batches", "Live"), ("Sales & bills", "Live"), ("Reorder plan", "Live"),
                                ("Alerts", "Live"), ("Shade catalog", "Live"), ("Visitor counts", "Live")]:
                html(f'<div class="apx-row" style="justify-content:space-between;padding:4px 0">'
                     f'<span>{name}</span><span style="color:#067647;font-weight:600">{state}</span></div>')
        if mode == "Gemini" and not has_key:
            reason = "Add a Gemini API key in Settings (sidebar)." if GENAI_AVAILABLE else \
                "The google-genai package isn't installed."
            html(f'<div class="apx-banner" style="background:#FDF1E2;display:block"><b>Gemini is not set up</b><br>'
                 f'<span class="apx-muted">{escape(reason)} Until then Aura answers offline.</span></div>')
        elif mode == "Gemini":
            html('<div class="apx-banner" style="background:#E5F3EB;display:block"><b>Gemini is on</b><br>'
                 '<span class="apx-muted">Answers any question in plain words, using only your shop data. '
                 'If Gemini fails, Aura falls back to offline answers and says so.</span></div>')
        else:
            html('<div class="apx-banner" style="background:#E7EDFB;display:block"><b>Offline mode</b><br>'
                 '<span class="apx-muted">Works without internet for common questions — stock, reorders, alerts, '
                 "today's numbers, colour suggestions.</span></div>")
        if SS.get("aura_history") and st.button("Clear chat", use_container_width=True):
            SS["aura_history"] = []
            SS.pop("aura_last_pick", None)
            st.rerun()
