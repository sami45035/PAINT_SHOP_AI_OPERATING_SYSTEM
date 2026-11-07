"""Colour Studio — show a customer their own room in any shade we stock."""
import hashlib
import io
from html import escape
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.colour_assistant import find_nearest_shades_by_hex, search_shades_by_query
from src.colour_studio import segmentation as seg
from src.colour_studio.quantity import best_packs, describe_packs, litres_needed
from src.colour_studio.recolour import outline_overlay, recolour, side_by_side
from src.colour_studio.recommend import current_wall_colour, recommend_shades, room_palette
from src.config import SAMPLE_ROOMS_DIR
from src.formatting import pack, rupees
from src.ui.theme import chip, html, page_header, show_image, show_table, step

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_CLICK = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CLICK = False

SAMPLES = {"Bedroom": "master_bedroom.jpg", "Living room": "modern_living_room.jpg", "Dining room": "cozy_dining_room.jpg"}
DISPLAY_W = 820
TEAL_RGB = (11, 110, 121)
SS = st.session_state


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner=False)
def _segmenter():
    path = seg.find_local_model()
    if not path:
        return None
    try:
        return seg.WallSegmenter.from_path(path)
    except Exception as exc:  # corrupt download, old onnxruntime, …
        SS["cs_model_error"] = f"The wall model couldn't start ({exc})."
        return None


def _ensure_model():
    if seg.find_local_model():
        return _segmenter()
    if SS.get("cs_download_failed"):
        return None
    bar = st.progress(0.0, text="Getting the wall model ready — one-time download…")
    path, msg = seg.download_model(lambda frac, text: bar.progress(min(max(frac, 0.0), 1.0), text=text))
    bar.empty()
    if not path:
        SS["cs_download_failed"] = msg
        return None
    _segmenter.clear()
    return _segmenter()


# ------------------------------------------------------------------ photo
def _load_photo():
    source = st.radio("Photo", ["Sample rooms", "Upload photo", "Take photo"], horizontal=True,
                      key="cs_source", label_visibility="collapsed")
    raw: Optional[bytes] = None
    name = ""
    if source == "Upload photo":
        f = st.file_uploader("Room photo (JPG or PNG)", type=["jpg", "jpeg", "png", "webp"], key="cs_upload")
        if f:
            raw, name = f.getvalue(), f.name
    elif source == "Take photo":
        f = st.camera_input("Take a photo of the wall", key="cs_camera")
        if f:
            raw, name = f.getvalue(), "camera.jpg"
    else:
        choice = st.selectbox("Sample room", list(SAMPLES), key="cs_sample", label_visibility="collapsed")
        path = SAMPLE_ROOMS_DIR / SAMPLES[choice]
        if path.exists():
            raw, name = path.read_bytes(), choice
    if raw is None:
        return None, None
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img_id = hashlib.md5(raw).hexdigest()[:12]
    return img_id, np.array(img)


def _reset_for(img_id: str) -> None:
    if SS.get("cs_img_id") != img_id:
        SS["cs_img_id"] = img_id
        SS["cs_taps"] = []
        SS["cs_sel"] = None          # None = pick a sensible default
        SS["cs_last_click"] = None
        SS["cs_render_cache"] = {}


# ------------------------------------------------------------------ segmentation
def _segmentation(img_id: str, image: np.ndarray, segmenter) -> seg.SegmentationResult:
    cache: Dict = SS.setdefault("cs_seg_cache", {})
    if segmenter is not None:
        key = ("model", img_id)
        if key not in cache:
            with st.spinner("Finding the walls…"):
                cache.clear()
                cache[key] = segmenter.segment(image)
        return cache[key]
    key = ("tap", img_id, tuple(SS.get("cs_taps", [])))
    if key not in cache:
        cache.clear()
        cache[key] = seg.tap_result(image, SS.get("cs_taps", []))
    return cache[key]


def _default_selection(result: seg.SegmentationResult) -> List[int]:
    if not result.regions:
        return []
    if result.method == "tap":
        return [r.id for r in result.regions]
    back = [r for r in result.regions if r.name.startswith("Back")]
    return [(back[0] if back else max(result.regions, key=lambda r: r.area_share)).id]


def _handle_click(click, result: seg.SegmentationResult, shown_w: int, shown_h: int) -> None:
    if not click:
        return
    sig = (click.get("x"), click.get("y"), click.get("unix_time"))
    if sig == SS.get("cs_last_click"):
        return
    SS["cs_last_click"] = sig
    w = click.get("width") or shown_w
    h = click.get("height") or shown_h
    x, y = click["x"] / w, click["y"] / h
    region = result.region_at(x, y)
    sel = set(SS.get("cs_sel") or [])
    if region is not None:
        sel.symmetric_difference_update({region.id})   # tap again to un-select
        SS["cs_sel"] = sorted(sel)
    elif result.method == "tap":
        SS.setdefault("cs_taps", []).append((x, y))   # grow a new wall from this point
        SS["cs_sel"] = None
    st.rerun()


# ------------------------------------------------------------------ shades
def _set_shade(shade: dict) -> None:
    SS["cs_shade"] = shade
    SS["cs_force_after"] = True   # switch the Before/After toggle on the next run


def _stock_chip(shade: dict) -> str:
    n = int(shade.get("total_stock", 0))
    if n <= 0:
        return chip("act_now", "Out of stock")
    if n < 15:
        return chip("soon", f"Low · {n}")
    return chip("healthy", "In stock")


def _shade_list(shades: List[dict], key: str, reason_key: str = "reason") -> None:
    current = (SS.get("cs_shade") or {}).get("hex_code")
    for i, s in enumerate(shades):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1.3], vertical_alignment="center")
            with c1:
                sub = s.get(reason_key) or f"{s.get('finish', '')} · {s.get('shade_code', '')}"
                html(
                    f'<div class="apx-row" style="padding:0"><span class="apx-sw" style="width:44px;height:44px;'
                    f'background:{escape(s["hex_code"])}"></span><span class="apx-row-main"><b>{escape(s["shade_name"])}</b>'
                    f' <span class="apx-muted">· {escape(s["brand"])}</span><span class="apx-row-sub">{escape(sub)}</span></span>'
                    f'{_stock_chip(s)}</div>'
                )
            with c2:
                selected = s["hex_code"] == current
                if st.button("Chosen" if selected else "Use", key=f"{key}_{i}", disabled=selected,
                             type="primary" if not selected else "secondary", use_container_width=True):
                    _set_shade(s)
                    st.rerun()


# ------------------------------------------------------------------ page
def render() -> None:
    page_header("Colour Studio", "Show a customer their own room in any shade we stock.")

    segmenter = _ensure_model()
    if segmenter is None:
        why = SS.get("cs_model_error") or SS.get("cs_download_failed") or "The wall model isn't available."
        html(f'<div class="apx-banner" style="background:#FDF1E2">{chip("soon", "Tap mode")}'
             f'<span>{escape(why)} Tap each wall on the photo to select it instead.</span></div>')

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            step(1, "Room photo")
            img_id, image = _load_photo()
        if image is None:
            st.info("Add a room photo to start — upload one, take one, or use a sample room.")
            return
        _reset_for(img_id)
        result = _segmentation(img_id, image, segmenter)
        if SS.get("cs_sel") is None:
            SS["cs_sel"] = _default_selection(result)
        alpha = result.selection_alpha(SS["cs_sel"])
        palette = room_palette(result.image, result.label_map, result.id2label, alpha > 0.5)
        wall_now = current_wall_colour(result.image, alpha)
        recs = recommend_shades(palette, wall_now, SS.get("cs_mood", ""), limit=5)
        if recs and not SS.get("cs_shade"):
            SS["cs_shade"] = recs[0]

        with st.container(border=True):
            h1, h2 = st.columns([3, 2], vertical_alignment="center")
            with h1:
                step(2, "Pick a wall")
            with h2:
                if SS.pop("cs_force_after", False) or "cs_mode_radio" not in SS:
                    SS["cs_mode_radio"] = "After"
                mode = st.radio("View", ["Before", "After"], horizontal=True, key="cs_mode_radio",
                                label_visibility="collapsed")

            names = {r.name: r.id for r in result.regions}
            if names:
                cols = st.columns(len(names) + 1)
                sel = set(SS["cs_sel"])
                for col, (name, rid) in zip(cols, names.items()):
                    with col:
                        on = rid in sel
                        if st.button(("✓ " if on else "") + name, key=f"cs_wall_{rid}", use_container_width=True,
                                     type="primary" if on else "secondary"):
                            sel.symmetric_difference_update({rid})
                            SS["cs_sel"] = sorted(sel)
                            st.rerun()
                with cols[-1]:
                    all_on = sel == set(names.values())
                    if st.button("All walls", key="cs_wall_all", use_container_width=True,
                                 type="primary" if all_on else "secondary"):
                        SS["cs_sel"] = [] if all_on else sorted(names.values())
                        st.rerun()

            shade = SS.get("cs_shade")
            base = result.image
            if mode == "After" and shade and SS["cs_sel"]:
                cache = SS.setdefault("cs_render_cache", {})
                rkey = (img_id, tuple(SS["cs_sel"]), shade["hex_code"], result.method, tuple(SS.get("cs_taps", [])))
                if rkey not in cache:
                    cache.clear()
                    cache[rkey] = recolour(base, alpha, shade["hex_code"])
                shown = cache[rkey]
            else:
                shown = outline_overlay(
                    base,
                    [(r.mask, TEAL_RGB, 0.28 if r.id in SS["cs_sel"] else 0.0) for r in result.regions],
                    thickness=2,
                )

            h, w = shown.shape[:2]
            disp_h = int(h * DISPLAY_W / w)
            pil = Image.fromarray(shown).resize((DISPLAY_W, disp_h), Image.LANCZOS)
            if HAS_CLICK:
                click = streamlit_image_coordinates(pil, key=f"cs_click_{img_id}")
                _handle_click(click, result, DISPLAY_W, disp_h)
            else:
                show_image(pil)
                st.caption("Install `streamlit-image-coordinates` to pick walls by tapping the photo.")

            if result.method == "model":
                msg = (f"<b>{len(result.regions)} wall{'s' if len(result.regions) != 1 else ''} found</b> in "
                       f"{result.seconds:.1f}s · ceiling, doors, windows and furniture are left untouched · tap a wall to add or remove it")
            elif result.regions:
                msg = f"<b>{len(result.regions)} wall{'s' if len(result.regions) != 1 else ''} picked</b> · tap another wall to add it, tap a painted wall to remove it"
            else:
                msg = "<b>Tap a wall on the photo</b> to select it"
            html(f'<div class="apx-muted">{msg}</div>')
            if result.method == "tap" and SS.get("cs_taps"):
                if st.button("Clear taps", key="cs_clear_taps"):
                    SS["cs_taps"], SS["cs_sel"] = [], None
                    st.rerun()

    with right:
        with st.container(border=True):
            step(3, "Choose a shade")
            tab_room, tab_search, tab_exact = st.tabs(["Suits this room", "Search", "Exact colour"])
            with tab_room:
                if palette:
                    sw = "".join(f'<span class="apx-sw" title="{escape(p["name"])}" style="width:20px;height:20px;'
                                 f'background:rgb{p["rgb"]}"></span>' for p in palette[:4])
                    names_txt = ", ".join(p["name"] for p in palette[:3])
                    html(f'<div class="apx-row" style="font-size:12px;color:#5A6472">Room colours: {sw}</div>'
                         f'<div class="apx-muted" style="margin:-4px 0 8px">{escape(names_txt)}</div>')
                st.text_input("Room or mood (optional)", key="cs_mood", placeholder="e.g. calm bedroom, bright kitchen")
                _shade_list(recs, "cs_rec")
            with tab_search:
                q = st.text_input("Describe the colour or room", key="cs_q", placeholder="warm beige, emerald, kids room…")
                if q:
                    found = search_shades_by_query(q, limit=6)
                    if found:
                        _shade_list(found, "cs_find", reason_key="_none")
                    else:
                        st.caption("No shades match that — try a colour name or a room.")
            with tab_exact:
                picked = st.color_picker("Pick any colour", value=(SS.get("cs_shade") or {}).get("hex_code", "#D9B48C"),
                                         key="cs_picker")
                st.caption("Closest shades we sell:")
                _shade_list(find_nearest_shades_by_hex(picked, limit=4), "cs_near", reason_key="_none")

    _product_panel(result, alpha)


def _product_panel(result: seg.SegmentationResult, alpha: np.ndarray) -> None:
    shade = SS.get("cs_shade")
    if not shade:
        return
    with st.container(border=True):
        a, b, c = st.columns([1.2, 1.2, 1.2], gap="large")
        with a:
            html(
                f'<div class="apx-row" style="align-items:flex-start;padding:0"><span class="apx-sw-lg" style="background:{escape(shade["hex_code"])}"></span>'
                f'<span class="apx-row-main"><span class="apx-display" style="font-size:22px;font-weight:600">{escape(shade["shade_name"])}</span>'
                f'<span class="apx-row-sub">{escape(shade["brand"])} · {escape(shade.get("shade_code", ""))} · {escape(shade.get("finish", ""))}</span>'
                f'<span class="apx-row-sub">{escape(shade.get("category", ""))} · {escape(shade["hex_code"])}</span></span></div>'
            )
            if not shade.get("in_stock", True) and shade.get("alternatives"):
                alt = shade["alternatives"][0]
                html(f'<div class="apx-banner" style="background:#FDF1E2;margin-top:10px">{chip("soon", "Out of stock")}'
                     f'<span>Closest in stock: <b>{escape(alt["brand"])} {escape(alt["shade_name"])}</b> ({int(alt["current_stock"])} cans)</span></div>')
                if st.button("Use this instead", key="cs_use_alt"):
                    found = search_shades_by_query(alt["shade_name"], limit=1)
                    if found:
                        _set_shade(found[0])
                        st.rerun()
        packs = shade.get("pack_options") or []
        with b:
            if packs:
                show_table(
                    [{"Pack": pack(p["size_litres"]), "Price": rupees(p["price"]), "On shelf": f'{p["stock"]} cans'} for p in packs]
                )
        with c:
            html('<div class="apx-label">How much paint?</div>')
            q1, q2, q3 = st.columns(3)
            width_ft = q1.number_input("Width (ft)", 1.0, 200.0, 12.0, 1.0, key="cs_w")
            height_ft = q2.number_input("Height (ft)", 1.0, 40.0, 10.0, 0.5, key="cs_h")
            coats = q3.number_input("Coats", 1, 4, 2, 1, key="cs_coats")
            need = litres_needed(width_ft, height_ft, int(coats), shade.get("category", "Interior Emulsion"))
            plan = best_packs(need["litres"], [{"size_litres": p["size_litres"], "price": p["price"],
                                                "stock": p["stock"], "sku": p["sku"]} for p in packs])
            if plan:
                html(f'<div style="font-size:15px">Needs about <b>{need["litres"]:.1f} L</b> → suggest <b>{escape(describe_packs(plan))}</b> '
                     f'({rupees(plan["total_price"])})</div>')
            else:
                html(f'<div style="font-size:15px">Needs about <b>{need["litres"]:.1f} L</b> — not enough on the shelf for this wall.</div>')
            st.caption(f"Assumes about {need['coverage']} sq ft per litre per coat for {shade.get('category', 'this paint').lower()}.")

            if SS.get("cs_sel"):
                after = recolour(result.image, alpha, shade["hex_code"])
                buf = io.BytesIO()
                Image.fromarray(side_by_side(result.image, after)).save(buf, format="JPEG", quality=90)
                st.download_button("Save before / after", buf.getvalue(), file_name=f"{shade['shade_name'].replace(' ', '_')}_preview.jpg",
                                   mime="image/jpeg", type="primary", use_container_width=True)
