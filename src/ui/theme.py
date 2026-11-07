"""
UI system v1 — colours, type and small HTML components shared by every screen.

Matches the wireframe: light theme, Apex Teal brand colour, status colours that
always come with a word, Space Grotesk headings and Inter body text.
"""
from html import escape
from typing import Optional

import streamlit as st

# Base colours
TEAL = "#0B6E79"
TEAL_DARK = "#08555E"
TEAL_WASH = "#E2F0F1"
INK = "#15202B"
SLATE = "#5A6472"
LINE = "#E2E5EA"
GROUND = "#F4F5F7"
SURFACE = "#FFFFFF"

# Chart colours (validated: chroma, contrast and colour-blind separation)
CHART_PRIMARY = "#00869C"
CHART_SECONDARY = "#D97A2B"

# Status: (text colour, background) — always shown with a word
STATUS = {
    "act_now": ("#B42318", "#FCEBE9", "Act now"),
    "soon": ("#B54708", "#FDF1E2", "Soon"),
    "healthy": ("#067647", "#E5F3EB", "Healthy"),
    "info": ("#1D4ED8", "#E7EDFB", "Info"),
    "look_into": ("#1D4ED8", "#E7EDFB", "Look into"),
    "neutral": ("#5A6472", "#F0F1F4", ""),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}
h1, h2, h3, h4, .apx-display {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-size: 28px !important; font-weight: 700 !important; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a {{ border-radius: 10px; min-height: 40px; }}

.apx-sub {{ color: {SLATE}; font-size: 15px; margin: -6px 0 18px; }}
.apx-brand {{ display: flex; align-items: center; gap: 10px; margin: 4px 0 12px; }}
.apx-mark {{ width: 36px; height: 36px; border-radius: 9px; background: {TEAL}; display: flex; align-items: flex-end;
  justify-content: center; gap: 3px; padding-bottom: 7px; box-sizing: border-box; flex: none; }}
.apx-mark span {{ width: 5px; border-radius: 2px; display: block; }}
.apx-brand-name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px; color: {INK}; line-height: 1.2; }}
.apx-brand-sub {{ font-size: 12px; color: {SLATE}; }}

.apx-pill {{ display: inline-flex; align-items: center; gap: 8px; height: 30px; padding: 0 12px; border-radius: 999px;
  font-size: 13px; font-weight: 600; }}
.apx-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.apx-chip {{ display: inline-flex; align-items: center; height: 24px; padding: 0 10px; border-radius: 999px;
  font-size: 12px; font-weight: 600; white-space: nowrap; }}

.apx-card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px; padding: 16px 18px; height: 100%; box-sizing: border-box; }}
.apx-card.attention {{ border-color: #F3C4BE; }}
.apx-label {{ font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: {SLATE}; }}
.apx-kpi {{ font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 600; color: {INK};
  font-variant-numeric: tabular-nums; line-height: 1.25; margin: 4px 0 2px; }}
.apx-kpi-sub {{ font-size: 13px; color: {SLATE}; }}
.apx-up {{ color: #067647; font-weight: 600; }}
.apx-down {{ color: #B42318; font-weight: 600; }}

.apx-banner {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-radius: 12px; font-size: 14px;
  color: {INK}; margin-bottom: 16px; }}

.apx-row {{ display: flex; align-items: center; gap: 12px; padding: 7px 0; font-size: 14px; color: {INK}; }}
.apx-sw {{ width: 28px; height: 28px; border-radius: 6px; border: 1px solid {LINE}; flex: none; }}
.apx-sw-lg {{ width: 84px; height: 84px; border-radius: 12px; border: 1px solid {LINE}; flex: none; }}
.apx-row-main {{ flex: 1; min-width: 0; }}
.apx-row-sub {{ display: block; font-size: 12px; color: {SLATE}; }}
.apx-row-right {{ font-variant-numeric: tabular-nums; font-weight: 600; }}

.apx-step {{ display: flex; align-items: center; gap: 10px; margin: 2px 0 8px; }}
.apx-step-n {{ width: 26px; height: 26px; border-radius: 50%; background: {INK}; color: #fff; font-size: 13px; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center; flex: none; }}
.apx-step-t {{ font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: {INK}; }}

.apx-foot {{ font-size: 12px; color: {SLATE}; line-height: 1.9; }}
.apx-foot b {{ color: {INK}; font-weight: 600; }}
.apx-muted {{ color: {SLATE}; font-size: 13px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        html(f'<div class="apx-sub">{escape(subtitle)}</div>')


def chip(level: str, text: Optional[str] = None) -> str:
    fg, bg, default = STATUS.get(level, STATUS["neutral"])
    return f'<span class="apx-chip" style="background:{bg};color:{fg}">{escape(text or default)}</span>'


def banner(level: str, chip_text: str, message_html: str) -> None:
    fg, bg, _ = STATUS.get(level, STATUS["info"])
    html(
        f'<div class="apx-banner" style="background:{bg}">'
        f'<span class="apx-chip" style="background:{fg};color:#fff">{escape(chip_text)}</span>'
        f"<span>{message_html}</span></div>"
    )


def kpi_card(label: str, value: str, sub_html: str = "", attention: bool = False, value_colour: Optional[str] = None) -> None:
    colour = f' style="color:{value_colour}"' if value_colour else ""
    html(
        f'<div class="apx-card{" attention" if attention else ""}">'
        f'<div class="apx-label">{escape(label)}</div>'
        f'<div class="apx-kpi"{colour}>{escape(value)}</div>'
        f'<div class="apx-kpi-sub">{sub_html}</div></div>'
    )


def change_html(pct: Optional[float], suffix: str = "vs yesterday") -> str:
    if pct is None:
        return f'<span class="apx-muted">No sales yesterday to compare</span>'
    cls, arrow = ("apx-up", "▲") if pct >= 0 else ("apx-down", "▼")
    return f'<span class="{cls}">{arrow} {abs(pct):.0f}% {suffix}</span>'


def swatch_row(hex_code: str, title: str, sub: str = "", right_html: str = "") -> str:
    return (
        f'<div class="apx-row"><span class="apx-sw" style="background:{escape(hex_code)}"></span>'
        f'<span class="apx-row-main">{escape(title)}<span class="apx-row-sub">{escape(sub)}</span></span>'
        f'<span class="apx-row-right">{right_html}</span></div>'
    )


def step(n: int, title: str) -> None:
    html(f'<div class="apx-step"><span class="apx-step-n">{n}</span><span class="apx-step-t">{escape(title)}</span></div>')


# --- Streamlit version compatibility --------------------------------------------
# Newer Streamlit uses width="stretch"; older versions use use_container_width=True.

def show_image(image, caption: Optional[str] = None) -> None:
    try:
        st.image(image, caption=caption, width="stretch")
    except Exception:
        st.image(image, caption=caption, use_container_width=True)


def show_chart(fig, key: Optional[str] = None) -> None:
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def show_table(df, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
