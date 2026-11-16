"""Shade Match — a customer brings a colour; find out exactly what it is and which shade to tint.

Asked for by the dealer we showed the app to: customers walk in with a paint chip, a cushion
or a photo and want "exactly this colour".
"""
import hashlib
import io
from html import escape
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src import clock
from src import colour_match as cm
from src.ui import nav
from src.ui.theme import banner, chip, html, page_header, show_image, step

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_CLICK = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CLICK = False

SS = st.session_state
SOURCES = ["Take photo", "Upload photo", "Colour code"]
TAP_MODES = ["Tap the colour", "Tap white paper"]
DISPLAY_W = 640
MAX_SIDE = 1600
BASE_WORDS = {1: "pastel base", 2: "light base", 3: "mid-tone base", 4: "deep base"}
VERDICT_LEVEL = {"exact": "healthy", "close": "info", "family": "soon", "none": "act_now"}


# ------------------------------------------------------------------ input
def _load() -> Tuple[Optional[str], Optional[np.ndarray], Optional[str]]:
    """(image id, image, typed hex) — one of image / hex is set."""
    source = st.radio("Sample", SOURCES, horizontal=True, key="sm_source", label_visibility="collapsed")
    raw = None
    if source == "Take photo":
        f = st.camera_input("Photo of the colour", key="sm_camera",
                            help="Fill the middle of the photo with the colour; put plain white paper beside it.")
        raw = f.getvalue() if f else None
    elif source == "Upload photo":
        f = st.file_uploader("Photo of the colour (JPG or PNG)", type=["jpg", "jpeg", "png", "webp"], key="sm_upload")
        raw = f.getvalue() if f else None
    else:
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        typed = c1.text_input("Colour code (hex)", key="sm_hex", placeholder="#C85A32",
                              help="From a website, Instagram post or another brand's shade card")
        picked = c2.color_picker("or pick", value="#C85A32", key="sm_pick")
        code = typed.strip() or picked
        if not cm.HEX_RE.match(code):
            st.caption("Type 6 hex digits, like #C85A32.")
            return None, None, None
        return None, None, "#" + cm.HEX_RE.match(code).group(1).upper()
    if raw is None:
        return None, None, None
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img_id = hashlib.md5(raw).hexdigest()[:12]
    if SS.get("sm_img_id") != img_id:              # a new photo: sample its middle, no white point yet
        SS["sm_img_id"], SS["sm_pt"], SS["sm_white"], SS["sm_last_click"] = img_id, (0.5, 0.5), None, None
        SS["sm_mode_next"] = TAP_MODES[0]
    return img_id, np.array(img), None


def _marked(image: np.ndarray, pt, white) -> Image.Image:
    shown = image.copy()
    h, w = shown.shape[:2]
    r = max(6, int(round(cm.SAMPLE_SHARE * min(h, w))))
    t = max(2, min(h, w) // 300)
    for spot, colour, label in ((white, (232, 103, 58), "W"), (pt, (255, 255, 255), None)):
        if spot is None:
            continue
        c = (int(spot[0] * (w - 1)), int(spot[1] * (h - 1)))
        cv2.circle(shown, c, r + t * 2, (20, 30, 35), t * 2, cv2.LINE_AA)
        cv2.circle(shown, c, r + t * 2, colour, t, cv2.LINE_AA)
        if label:
            cv2.putText(shown, label, (c[0] + r + t * 4, c[1] + r // 2), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, r / 18),
                        colour, max(2, t), cv2.LINE_AA)
    disp_h = int(h * DISPLAY_W / w)
    return Image.fromarray(shown).resize((DISPLAY_W, disp_h), Image.LANCZOS)


def _photo_panel(img_id: str, image: np.ndarray) -> None:
    if "sm_mode_next" in SS:
        SS["sm_mode"] = SS.pop("sm_mode_next")
    mode = st.radio("Tap to", TAP_MODES, horizontal=True, key="sm_mode", label_visibility="collapsed")
    pil = _marked(image, SS.get("sm_pt"), SS.get("sm_white"))
    if HAS_CLICK:
        click = streamlit_image_coordinates(pil, key=f"sm_click_{img_id}")
        if click:
            sig = (click.get("x"), click.get("y"), click.get("unix_time"))
            if sig != SS.get("sm_last_click"):
                SS["sm_last_click"] = sig
                fx = min(max(click["x"] / (click.get("width") or pil.width), 0.0), 1.0)
                fy = min(max(click["y"] / (click.get("height") or pil.height), 0.0), 1.0)
                if mode == TAP_MODES[1]:
                    SS["sm_white"] = (fx, fy)
                    SS["sm_mode_next"] = TAP_MODES[0]
                else:
                    SS["sm_pt"] = (fx, fy)
                st.rerun()
    else:
        show_image(pil)
    tip = ("Now tap the plain white paper in the photo — it corrects the shop's lighting."
           if mode == TAP_MODES[1] else
           "Tap the colour to read it (white ring). For an exact reading, also tap white paper (orange W).")
    html(f'<div class="apx-muted">{tip}</div>')
    if SS.get("sm_white") is not None and st.button("Remove white point", key="sm_clear_white"):
        SS["sm_white"] = None
        st.rerun()


# ------------------------------------------------------------------ results
def _reading(rgb, sample: Optional[cm.Sample]) -> None:
    hex_code = cm.rgb_to_hex(rgb)
    name, name_de = cm.nearest_name(rgb)
    plain = cm.describe(rgb)
    n = cm.base_number(rgb)
    html(
        f'<div class="apx-row" style="gap:16px;padding:4px 0 10px"><span class="apx-sw-lg" style="background:{hex_code}"></span>'
        f'<span class="apx-row-main"><span style="font-family:Space Grotesk,Inter,sans-serif;font-size:30px;'
        f'font-weight:700;letter-spacing:.5px">{hex_code}</span>'
        f'<span class="apx-row-sub" style="font-size:15px;color:#15202B">{escape(plain.capitalize())}</span>'
        f'<span class="apx-row-sub">{"" if name_de < 1 else "close to "}<b>{escape(name)}</b> (standard colour name)</span>'
        f'</span></div>')
    r, g, b = (int(round(v)) for v in rgb)
    c1, c2, c3 = st.columns(3)
    c1.metric("RGB", f"{r}, {g}, {b}")
    c2.metric("Light reflectance", f"{cm.lrv(rgb):.0f}", help="LRV: 0 = black, 100 = white. Below ~15 a room feels dark.")
    c3.metric("Tint on", f"Base {n}", help=f"Base {n} = {BASE_WORDS[n]}: the lighter the colour, the more white in the base.")
    if sample is not None:
        if sample.corrected:
            html(f'<div style="margin:4px 0">{chip("healthy", "Lighting corrected with white paper")}</div>')
        else:
            html(f'<div style="margin:4px 0">{chip("soon", "Lighting not corrected")} '
                 f'<span class="apx-muted">tap white paper in the photo for an exact reading</span></div>')
        for w in sample.warnings:
            banner("soon", "Check", escape(w))


def _try_on_wall(shade: dict) -> None:
    from src.colour_assistant import find_nearest_shades_by_hex

    found = find_nearest_shades_by_hex(shade["hex"], limit=1)
    if found:
        SS["cs_shade"] = found[0]
        SS["cs_look"] = None
        SS["cs_force_after"] = True
    nav.go("colour")


def _matches(rgb) -> None:
    found = cm.match(rgb, limit=4)
    if not found:
        st.caption("No shades in the catalogue yet.")
        return
    best = found[0]
    n = cm.base_number(rgb)
    if best["verdict_key"] in ("exact", "close"):
        where = (f" · tint on <b>{escape(best['base_code'])}</b> ({best['stock']} in stock)" if best.get("base_code")
                 else f" · {best['stock']} in stock" if best.get("stock") is not None else " · from a brand fan deck")
        banner("healthy" if best["verdict_key"] == "exact" else "info", best["verdict"],
               f"<b>{escape(best['name'])}</b> · {escape(best['brand'])} {escape(best['code'])}{where}"
               + ("" if best["verdict_key"] == "exact" else " — show the customer the shade card to confirm"))
    else:
        bases = cm.bases_in_stock(n)
        have = ", ".join(f"{b['code']} ({b['cans']})" for b in bases[:4]) or "none in stock"
        banner("soon", "Tint to order",
               f"Not on our shade cards. Make it on a <b>Base {n}</b> ({BASE_WORDS[n]}) with the tinting machine "
               f"— Base {n} cans in stock: {escape(have)}.")

    for i, s in enumerate(found):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1.6], vertical_alignment="center")
            with c1:
                label = f"{s['verdict'].split(' —')[0]} · ΔE {s['de']:.1f}"
                stock = ("" if s.get("stock") is None else
                         f" · {s['base_code'] + ' · ' if s.get('base_code') else ''}{s['stock']} in stock")
                html(
                    f'<div class="apx-row" style="padding:0"><span style="display:flex;flex:none">'
                    f'<span class="apx-sw" style="width:36px;height:44px;border-radius:6px 0 0 6px;background:{cm.rgb_to_hex(rgb)}"></span>'
                    f'<span class="apx-sw" style="width:36px;height:44px;border-radius:0 6px 6px 0;border-left:none;background:{escape(s["hex"])}"></span></span>'
                    f'<span class="apx-row-main"><b>{escape(s["name"])}</b> <span class="apx-muted">· {escape(s["brand"])}</span>'
                    f'<span class="apx-row-sub">{escape(s["code"])} · {escape(s["hex"])}{escape(stock)}</span></span>'
                    f'{chip(VERDICT_LEVEL[s["verdict_key"]], label)}</div>')
            with c2:
                if s["source"] == "shop" and st.button("Try on a wall", key=f"sm_try_{i}", use_container_width=True):
                    _try_on_wall(s)
    st.caption("ΔE = how different two colours look (CIEDE2000). Under 2 most people can't tell them apart; "
               "above 5 the difference is clear side by side. Left half of each swatch = the customer's colour.")
    _ticket(rgb, best)


def _ticket(rgb, best: dict) -> None:
    hex_code = cm.rgb_to_hex(rgb)
    n = cm.base_number(rgb)
    if best["verdict_key"] in ("exact", "close"):
        use = f"{best['brand']} {best['name']} ({best['code']})" + (f" on {best['base_code']}" if best.get("base_code") else "")
    else:
        use = f"custom tint on Base {n}"
    text = (f"Shade match · {clock.now():%d %b %Y, %I:%M %p}\n"
            f"Customer colour: {hex_code} ({cm.describe(rgb)})\n"
            f"Use: {use}")
    with st.expander("Ticket for the tinting counter"):
        st.code(text, language=None)


def _fan_decks() -> None:
    with st.expander("Brand shade cards (fan decks)"):
        have = cm.library_summary()
        if have:
            st.caption("Also matching against: " + ", ".join(f"{b} ({n})" for b, n in have))
        else:
            st.caption("Load a brand's shade card to match against every shade it makes, not only the ones on "
                       "your shelves. A CSV with columns brand, code, name, hex.")
        f = st.file_uploader("Shade card CSV", type=["csv"], key="sm_deck")
        if f is not None and SS.get("sm_deck_done") != f.file_id:
            rows, problems = cm.parse_fan_deck(f.getvalue())
            if rows:
                cm.save_fan_deck(rows)
                st.success(f"Loaded {len(rows)} shades.")
            for p in problems[:5]:
                st.caption(p)
            SS["sm_deck_done"] = f.file_id
        c1, c2 = st.columns(2)
        c1.download_button("CSV template", cm.FAN_DECK_TEMPLATE, file_name="shade_card_template.csv",
                           mime="text/csv", use_container_width=True)
        if have and c2.button("Remove loaded shade cards", key="sm_deck_clear", use_container_width=True):
            cm.clear_fan_decks()
            st.rerun()


# ------------------------------------------------------------------ page
def render() -> None:
    page_header("Shade Match", "A customer brings a colour — a paint chip, a cushion, a photo. "
                               "Find out exactly what it is and which shade to tint.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            step(1, "The customer's colour")
            img_id, image, typed = _load()
            if image is not None:
                _photo_panel(img_id, image)
            elif typed is None:
                html('<div class="apx-muted" style="line-height:1.6">For the best reading: hold the sample flat, '
                     'out of direct sun, with <b>plain white paper beside it</b>, and fill the middle of the photo '
                     'with the colour.</div>')
    if image is None and typed is None:
        with right:
            _fan_decks()
        return

    sample = None
    if image is not None:
        fx, fy = SS.get("sm_pt") or (0.5, 0.5)
        sample = cm.sample_colour(image, fx, fy, white=SS.get("sm_white"))
        rgb = sample.rgb
    else:
        rgb = tuple(float(v) for v in cm.hex_to_rgb(typed))

    with right:
        with st.container(border=True):
            step(2, "What colour it is")
            _reading(rgb, sample)
        with st.container(border=True):
            step(3, "Closest shades we can make")
            _matches(rgb)
        _fan_decks()
