"""Colour Studio — show a customer their own room in any shade we stock."""
import hashlib
import io
from html import escape
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.colour_assistant import find_nearest_shades_by_hex, search_shades_by_query
from src.colour_studio import segmentation as seg
from src.colour_studio.quantity import best_packs, describe_packs, litres_needed
from src.colour_studio.recolour import blend, light_map, outline_overlay, paint_layer, side_by_side
from src.colour_studio import palettes as pal
from src.colour_studio.recommend import current_wall_colour, enrich_wall_paints, recommend_shades, room_palette
from src.config import SAMPLE_ROOMS_DIR
from src.formatting import pack, rupees
from src.ui.components import apply_edits, decode_edits, to_data_url, wall_brush
from src.ui.theme import chip, html, page_header, show_image, show_table, step

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_CLICK = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CLICK = False

SAMPLES = {"Bedroom": "master_bedroom.jpg", "Living room": "modern_living_room.jpg", "Dining room": "cozy_dining_room.jpg"}
VIEW_MODES = ["Pick walls", "Touch up with brush"]
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
        SS["cs_cache"], SS["cs_cache_sizes"] = {}, {}   # everything computed for this photo (see _memo)
        SS["cs_edits_url"] = ""      # brush touch-ups (PNG: red = painted, green = erased)
        SS["cs_edits"] = None
        SS["cs_edits_ts"] = 0
        SS["cs_brush_reset"] = SS.get("cs_brush_reset", 0) + 1
        SS["cs_view"] = VIEW_MODES[0]
        SS["cs_look"], SS["cs_look_plan"], SS["cs_look_paints"] = None, None, None


CACHE_MB = 120          # per visitor: enough for every look, brush and preview of one photo


def _size(v) -> int:
    if isinstance(v, np.ndarray):
        return v.nbytes
    if isinstance(v, (bytes, str)):
        return len(v)
    if isinstance(v, (list, tuple)):
        return sum(_size(x) for x in v)
    if isinstance(v, dict):
        return sum(_size(x) for x in v.values())
    if isinstance(v, Image.Image):
        return v.width * v.height * len(v.getbands())
    return 1024


def _memo(key, fn, keep: int = 60):
    """Small per-photo cache, so a click never recomputes what it didn't change (kept under CACHE_MB)."""
    cache: Dict = SS.setdefault("cs_cache", {})
    sizes: Dict = SS.setdefault("cs_cache_sizes", {})
    if key in cache:
        cache[key] = cache.pop(key)      # mark as recently used
        return cache[key]
    val = fn()
    cache[key] = val
    sizes[key] = _size(val)
    while len(cache) > 1 and (len(cache) > keep or sum(sizes.get(k, 0) for k in cache) > CACHE_MB * 2**20):
        old = next(iter(cache))           # least recently used
        cache.pop(old)
        sizes.pop(old, None)
    return val


# ------------------------------------------------------------------ segmentation
def _segmentation(img_id: str, image: np.ndarray, segmenter) -> seg.SegmentationResult:
    cache: Dict = SS.setdefault("cs_seg_cache", {})
    if segmenter is not None:
        key = ("model", img_id)
        if key not in cache:
            with st.spinner("Finding the walls…"):
                cache.clear()
                cache[key] = segmenter.segment(image)
        base = cache[key]
    else:
        key = ("tap", img_id, tuple(SS.get("cs_taps", [])))
        if key not in cache:
            cache.clear()
            cache[key] = seg.tap_result(image, SS.get("cs_taps", []))
        base = cache[key]
    return base


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
def _set_shade(shade: dict, keep_look: bool = False) -> None:
    SS["cs_shade"] = shade
    if not keep_look:
        SS["cs_look"] = None          # a single shade replaces a designer look
    SS["cs_force_after"] = True   # switch "Show the new colour" back on for the next run


def _scheme(shade: Optional[dict]):
    """What's on the walls: one shade (its hex) or a look (wall id -> hex), as a hashable key."""
    look = SS.get("cs_look")
    if look and SS.get("cs_look_plan"):
        return ("look", look, tuple(sorted(SS["cs_look_plan"].items())))
    return (shade or {}).get("hex_code")


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
                if st.button("✓" if selected else "Use", key=f"{key}_{i}", disabled=selected,
                             help="On the photo now" if selected else "Show this shade on the wall",
                             type="primary" if not selected else "secondary", use_container_width=True):
                    _set_shade(s)
                    st.rerun()


# ------------------------------------------------------------------ cached pieces
def _recommendations(result: seg.SegmentationResult, alpha: np.ndarray, mood: str):
    palette = room_palette(result.image, result.label_map, result.id2label, alpha > 0.5)
    wall_now = current_wall_colour(result.image, alpha)
    return palette, wall_now, recommend_shades(palette, wall_now, mood, limit=5)


def _owner(result: seg.SegmentationResult) -> np.ndarray:
    """For every pixel, the index of the nearest wall region (so soft edges and brushed patches get a wall's colour)."""
    def build():
        if not result.regions:
            return np.zeros(result.image.shape[:2], np.int16)
        dists = [cv2.distanceTransform((~r.mask).astype(np.uint8), cv2.DIST_L2, 3) for r in result.regions]
        return np.argmin(np.stack(dists), axis=0).astype(np.int16)
    return _memo(("owner",), build)


def _painted(result: seg.SegmentationResult, sel: tuple, scheme, base_alpha: np.ndarray) -> np.ndarray:
    """The whole photo in this paint (or look), lit like the chosen walls (light map shared by every shade)."""
    light = _memo(("light", sel), lambda: light_map(result.image, base_alpha))
    if isinstance(scheme, str) or scheme is None:
        return _memo(("painted", sel, scheme), lambda: paint_layer(light, scheme or "#FFFFFF"))
    plan = dict(scheme[2])

    def build():
        owner = _owner(result)
        colours = list(dict.fromkeys(plan.values()))
        main = max(colours, key=lambda c: sum(1 for v in plan.values() if v == c))
        out = _memo(("painted", sel, main), lambda: paint_layer(light, main)).copy()
        idx = {r.id: i for i, r in enumerate(result.regions)}
        for rid, hx in plan.items():
            if hx != main and rid in idx:
                layer = _memo(("painted", sel, hx), lambda hx=hx: paint_layer(light, hx))
                m = owner == idx[rid]
                out[m] = layer[m]
        return out
    return _memo(("painted_scheme", sel, scheme), build)


def _after(result, sel, ets, scheme, base_alpha, alpha) -> np.ndarray:
    return _memo(("after", sel, ets, scheme),
                 lambda: blend(result.image, _painted(result, sel, scheme, base_alpha), alpha))


def _pick_view(img_id, result: seg.SegmentationResult, alpha, sel, ets, shade) -> None:
    if SS.pop("cs_force_after", False) or "cs_show_after" not in SS:
        SS["cs_show_after"] = True
    show_after = st.toggle("Show the new colour", key="cs_show_after")
    scheme = _scheme(shade)
    if show_after and scheme and sel:
        base_alpha = _memo(("base_alpha", sel), lambda: result.selection_alpha(list(sel)))
        shown = _after(result, sel, ets, scheme, base_alpha, alpha)
        tag = ("after", sel, ets, scheme)
    else:
        shown = _memo(("outline", sel), lambda: outline_overlay(
            result.image, [(r.mask, TEAL_RGB, 0.28 if r.id in sel else 0.0) for r in result.regions], thickness=2))
        tag = ("outline", sel)
    h, w = shown.shape[:2]
    disp_h = int(h * DISPLAY_W / w)
    pil = _memo(("display",) + tag, lambda: Image.fromarray(shown).resize((DISPLAY_W, disp_h), Image.LANCZOS))
    if HAS_CLICK:
        click = streamlit_image_coordinates(pil, key=f"cs_click_{img_id}")
        _handle_click(click, result, DISPLAY_W, disp_h)
    else:
        show_image(pil)
        st.caption("Install `streamlit-image-coordinates` to pick walls by tapping the photo.")

    touched = " · brush touch-ups applied" if SS.get("cs_edits") is not None else ""
    if result.method == "model":
        msg = (f"<b>{len(result.regions)} wall{'s' if len(result.regions) != 1 else ''} found</b> in "
               f"{result.seconds:.1f}s · tap a wall to add or remove it · missed a patch? switch to "
               f"<b>Touch up with brush</b>" + touched)
    elif result.regions:
        msg = (f"<b>{len(result.regions)} wall{'s' if len(result.regions) != 1 else ''} picked</b> · tap another wall to add it, "
               f"tap a painted wall to remove it" + touched)
    else:
        msg = "<b>Tap a wall on the photo</b> to select it"
    html(f'<div class="apx-muted">{msg}</div>')
    if result.method == "tap" and SS.get("cs_taps"):
        if st.button("Clear taps", key="cs_clear_taps"):
            SS["cs_taps"], SS["cs_sel"] = [], None
            st.rerun()


def _touch_up(img_id, result: seg.SegmentationResult, base_alpha, sel, shade) -> None:
    """Brush paint on / off in the browser — instant, no round trip per stroke."""
    h, w = result.image.shape[:2]
    orig_url = _memo(("orig_url",), lambda: to_data_url(result.image, "JPEG", 90))
    scheme = _scheme(shade)
    painted_url = _memo(("painted_url", sel, scheme),
                        lambda: to_data_url(_painted(result, sel, scheme, base_alpha), "JPEG", 90))
    base_url = _memo(("base_url", sel),
                     lambda: to_data_url((np.clip(base_alpha, 0, 1) * 255).astype(np.uint8), "PNG"))
    val = wall_brush(key=f"cs_brush_{img_id}_{SS.get('cs_brush_reset', 0)}", photo_key=img_id,
                     orig_url=orig_url, painted_url=painted_url, base_url=base_url, width=w, height=h,
                     edits_url=SS.get("cs_edits_url", ""), reset=SS.get("cs_brush_reset", 0))
    if val and val.get("key") == img_id and val.get("ts") and val["ts"] != SS.get("cs_edits_ts"):
        SS["cs_edits_url"] = val.get("edits") or ""
        SS["cs_edits"] = decode_edits(SS["cs_edits_url"], (h, w))
        SS["cs_edits_ts"] = val["ts"]
        st.rerun()                      # update shade ideas, paint amount and the download
    html('<div class="apx-muted"><b>Brush</b> paints a patch the walls missed · <b>Erase</b> takes paint off the '
         'headboard, ceiling or lamps · <b>[</b> <b>]</b> or Alt + scroll change the size · hold 👁 to compare</div>')


# ------------------------------------------------------------------ page
def render() -> None:
    page_header("Colour Studio", "Show a customer their own room in any shade we stock — or a complete designer look.")

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
        sel = tuple(SS["cs_sel"])
        ets = SS.get("cs_edits_ts", 0)
        base_alpha = _memo(("base_alpha", sel), lambda: result.selection_alpha(list(sel)))
        alpha = _memo(("alpha", sel, ets), lambda: apply_edits(base_alpha, SS.get("cs_edits")))
        palette, wall_now, recs = _memo(
            ("recs", sel, ets, SS.get("cs_mood", "")),
            lambda: _recommendations(result, alpha, SS.get("cs_mood", "")))
        if recs and not SS.get("cs_shade"):
            SS["cs_shade"] = recs[0]
        shade = SS.get("cs_shade")

        with st.container(border=True):
            h1, h2 = st.columns([3, 2], vertical_alignment="center")
            with h1:
                step(2, "Pick a wall")
            with h2:
                view = st.radio("Mode", VIEW_MODES, horizontal=True, key="cs_view", label_visibility="collapsed")
            touch_up = view == VIEW_MODES[1]

            names = {r.name: r.id for r in result.regions}
            if names:
                cols = st.columns(len(names) + 1)
                chosen = set(sel)
                for col, (name, rid) in zip(cols, names.items()):
                    with col:
                        on = rid in chosen
                        if st.button(("✓ " if on else "") + name, key=f"cs_wall_{rid}", use_container_width=True,
                                     type="primary" if on else "secondary"):
                            chosen.symmetric_difference_update({rid})
                            SS["cs_sel"] = sorted(chosen)
                            st.rerun()
                with cols[-1]:
                    all_on = chosen == set(names.values())
                    if st.button("All walls", key="cs_wall_all", use_container_width=True,
                                 type="primary" if all_on else "secondary"):
                        SS["cs_sel"] = [] if all_on else sorted(names.values())
                        st.rerun()

            if touch_up and shade:
                _touch_up(img_id, result, base_alpha, sel, shade)
            else:
                _pick_view(img_id, result, alpha, sel, ets, shade)

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

    _looks_section(img_id, result, palette, wall_now, ets)
    _product_panel(result, alpha, sel, ets)


# ------------------------------------------------------------------ designer looks
@st.cache_data(ttl=60, show_spinner=False)
def _stocked_paints():
    return enrich_wall_paints()


def _look_paints(look: dict) -> dict:
    """Each paint in a look -> a shade we can sell (stocked match, or house range tinted to order)."""
    stocked = _stocked_paints()
    return {role: pal.match_paint(p["name"], p["hex"], stocked) for role, p in look["paints"].items()}


def _thumb(result: seg.SegmentationResult, look: dict, ets: int, width: int = 420) -> np.ndarray:
    """The customer's room in this look, small: feature colour on the feature wall, main colour elsewhere."""
    all_ids = tuple(r.id for r in result.regions)

    def build():
        h, w = result.image.shape[:2]
        sw, sh = width, int(h * width / w)
        small = cv2.resize(result.image, (sw, sh), interpolation=cv2.INTER_AREA)
        base = _memo(("base_alpha", all_ids), lambda: result.selection_alpha(list(all_ids)))
        alpha = apply_edits(base, SS.get("cs_edits"))
        a_small = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        light = _memo(("light", all_ids), lambda: light_map(result.image, base))
        l_small = cv2.resize(light, (sw, sh), interpolation=cv2.INTER_AREA)
        owner = cv2.resize(_owner(result), (sw, sh), interpolation=cv2.INTER_NEAREST)
        plan = pal.wall_plan(result.regions, look)
        main = look["paints"]["walls"]["hex"]
        out = paint_layer(l_small, main)
        for i, r in enumerate(result.regions):
            hx = plan.get(r.id, main)
            if hx != main:
                layer = paint_layer(l_small, hx)
                m = owner == i
                out[m] = layer[m]
        return blend(small, out, a_small)
    return _memo(("thumb", look["id"], ets, width), build)


def _use_look(result: seg.SegmentationResult, look: dict) -> None:
    SS["cs_look"] = look["id"]
    SS["cs_look_plan"] = pal.wall_plan(result.regions, look)
    SS["cs_sel"] = sorted(r.id for r in result.regions)
    paints = _look_paints(look)
    SS["cs_look_paints"] = paints
    SS["cs_look_role"] = "walls"
    _set_shade(paints["walls"], keep_look=True)


def _swatches(look: dict, size: int = 26) -> str:
    items = [("Walls", look["paints"]["walls"]), ("Feature wall", look["paints"]["feature"]),
             ("Trim & ceiling", look["paints"]["trim"])] + [("Accent", a) for a in look["accents"]]
    flex = [3, 2, 1, 1, 1]
    return ('<div style="display:flex;height:%dpx;border-radius:8px;overflow:hidden;border:1px solid #E2E5EA;margin:8px 0">' % size
            + "".join(f'<span title="{escape(role)}: {escape(p["name"])}" style="flex:{f};background:{escape(p["hex"])}"></span>'
                      for (role, p), f in zip(items, flex)) + "</div>")


def _looks_section(img_id: str, result: seg.SegmentationResult, palette, wall_now, ets: int) -> None:
    if not result.regions:
        return
    with st.container(border=True):
        h1, h2 = st.columns([3, 2], vertical_alignment="bottom")
        with h1:
            step(4, "Designer looks for this room")
        with h2:
            st.text_input("Room or mood", key="cs_look_mood", placeholder="e.g. cozy, calm bedroom, festive",
                          label_visibility="collapsed")
        styles = st.pills("Style", pal.STYLES, selection_mode="multi", key="cs_styles", label_visibility="collapsed") or []
        text = f"{SS.get('cs_look_mood', '')} {SS.get('cs_mood', '')}"
        ctx = _memo(("look_ctx", ets, text), lambda: pal.room_context(
            result.image, palette, wall_now, result.label_map, result.id2label, len(result.regions), text))
        tone = {"warm": "warm wood tones", "cool": "cool tones", "neutral": "neutral tones"}[ctx["undertone"]]
        light = {"dim": "a dim room — lighter walls help", "bright": "plenty of light", "normal": "normal light"}[ctx["light"]]
        room = pal.ROOM_NAMES.get(ctx["room"], "room") if ctx["room"] else "room"
        html(f'<div class="apx-muted" style="margin:-4px 0 10px">Reading the photo: looks like a <b>{escape(room)}</b> · '
             f'{escape(tone)} · {escape(light)}. Each look is a full scheme: walls (60%), a feature wall (30%), '
             f'trim & ceiling (10%) and two decor accents.</div>')
        picks = _memo(("looks", ets, text, tuple(styles)), lambda: pal.recommend_looks(ctx, list(styles), text, limit=6))
        active = SS.get("cs_look")
        for row in range(0, len(picks), 3):
            cols = st.columns(3, gap="medium")
            for col, lk in zip(cols, picks[row:row + 3]):
                with col:
                    with st.container(border=True):
                        show_image(Image.fromarray(_thumb(result, lk, ets)))
                        tags = " ".join(chip("neutral", t) for t in lk["styles"][:2])
                        html(f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px">'
                             f'<span class="apx-display" style="font-size:17px;font-weight:600">{escape(lk["name"])}</span>'
                             f'<span>{tags}</span></div>{_swatches(lk)}'
                             f'<div class="apx-muted" style="line-height:1.45;min-height:3.9em">{escape(lk["mood"])}</div>'
                             f'<div style="font-size:12px;color:#067647;margin:6px 0 2px">✓ {escape(lk["reason"])}</div>')
                        on = active == lk["id"]
                        if st.button("✓ On the photo" if on else "Use this look", key=f"cs_look_{lk['id']}",
                                     type="secondary" if on else "primary", disabled=on, use_container_width=True):
                            _use_look(result, lk)
                            st.rerun()
        if active:
            _look_panel(result, active)


def _look_panel(result: seg.SegmentationResult, look_id: str) -> None:
    look = next((lk for lk in pal.looks() if lk["id"] == look_id), None)
    paints = SS.get("cs_look_paints")
    if look is None or not paints:
        return
    st.write("")
    a, b = st.columns([3, 2], gap="large")
    with a:
        html(f'<div class="apx-label">Your look · {escape(look["name"])}</div>')
        rows = []
        for role, label in (("walls", "Walls (60%)"), ("feature", "Feature wall (30%)"), ("trim", "Trim & ceiling (10%)")):
            s = paints[role]
            where = "tinted at the counter" if s.get("tinted") else (
                f"in stock · {int(s.get('total_stock', 0))} cans" if int(s.get("total_stock", 0)) > 0 else "out of stock")
            rows.append(f'<div class="apx-row"><span class="apx-sw" style="width:36px;height:36px;background:{escape(s["hex_code"])}"></span>'
                        f'<span class="apx-row-main"><b>{escape(s["shade_name"])}</b> <span class="apx-muted">· {escape(label)}</span>'
                        f'<span class="apx-row-sub">{escape(s["brand"])} · {escape(s.get("shade_code", ""))} · {escape(where)}</span></span></div>')
        acc = " · ".join(f'<span class="apx-sw" style="display:inline-block;width:14px;height:14px;vertical-align:-2px;background:{escape(x["hex"])}"></span> '
                         f'{escape(x["name"])}' for x in look["accents"])
        html("".join(rows) + f'<div class="apx-muted" style="margin-top:6px">Decor accents (cushions, curtains, a vase): {acc}</div>')
        if any(paints[r].get("tinted") for r in paints):
            st.caption("House-range colours (APX codes) are sample shades mixed on the tinting machine; "
                       "the shop's own stocked shade is used whenever one is a close match.")
    with b:
        regions = {r.name: r.id for r in result.regions}
        plan = SS.get("cs_look_plan") or {}
        feat_hex = look["paints"]["feature"]["hex"]
        current = next((n for n, rid in regions.items() if plan.get(rid) == feat_hex), "None")
        choice = st.selectbox("Feature wall", list(regions) + ["None"], index=(list(regions) + ["None"]).index(current),
                              key=f"cs_feature_{look_id}", help="Which wall gets the stronger colour")
        if choice != current:
            main = look["paints"]["walls"]["hex"]
            SS["cs_look_plan"] = {rid: (feat_hex if n == choice else main) for n, rid in regions.items()}
            st.rerun()
        role = st.radio("Plan paint for", ["walls", "feature", "trim"], horizontal=True, key="cs_look_role",
                        format_func={"walls": "Walls", "feature": "Feature wall", "trim": "Trim"}.get)
        if paints[role]["hex_code"] != (SS.get("cs_shade") or {}).get("hex_code"):
            _set_shade(paints[role], keep_look=True)
            st.rerun()
        board = _memo(("board", look_id, SS.get("cs_edits_ts", 0), tuple(sorted(plan.items()))),
                      lambda: _look_board(result, look, paints))
        st.download_button("Save look board", board, file_name=f"{look['id']}_look.png", mime="image/png",
                           use_container_width=True, help="A Pinterest-style card of this room in the look, with shade codes")


def _look_board(result: seg.SegmentationResult, look: dict, paints: dict) -> bytes:
    """Pinterest-style pin: the customer's room in the look, the swatches with names and codes."""
    from PIL import ImageDraw, ImageFont

    def font(size, bold=False):
        for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "Arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    W = 1000
    room = Image.fromarray(_thumb(result, look, SS.get("cs_edits_ts", 0), width=W))
    pad, sw_h = 40, 150
    H = room.height + 440
    board = Image.new("RGB", (W, H), (250, 248, 244))
    board.paste(room, (0, 0))
    d = ImageDraw.Draw(board)
    y = room.height + 30
    d.text((pad, y), look["name"], fill=(21, 32, 43), font=font(46, True))
    y += 62
    d.text((pad, y), look["mood"], fill=(90, 100, 114), font=font(22))
    y += 50
    items = [("WALLS", paints["walls"]), ("FEATURE", paints["feature"]), ("TRIM", paints["trim"])]
    items += [("ACCENT", {"hex_code": a["hex"], "shade_name": a["name"], "shade_code": "decor"}) for a in look["accents"]]
    cw = (W - 2 * pad - 4 * 14) // 5
    for i, (role, s) in enumerate(items):
        x = pad + i * (cw + 14)
        d.rounded_rectangle([x, y, x + cw, y + sw_h], radius=14, fill=tuple(int(s["hex_code"][j:j + 2], 16) for j in (1, 3, 5)),
                            outline=(226, 229, 234), width=2)
        d.text((x, y + sw_h + 12), role, fill=(90, 100, 114), font=font(15, True))
        d.text((x, y + sw_h + 34), s["shade_name"][:18], fill=(21, 32, 43), font=font(19, True))
        d.text((x, y + sw_h + 60), str(s.get("shade_code", ""))[:18], fill=(90, 100, 114), font=font(16))
    d.text((pad, H - 46), "Apex Colors & Paints · Colour Studio", fill=(11, 110, 121), font=font(20, True))
    buf = io.BytesIO()
    board.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _product_panel(result: seg.SegmentationResult, alpha: np.ndarray, sel: tuple = (), ets: int = 0) -> None:
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
                    [{"Pack": pack(p["size_litres"]), "Price": rupees(p["price"]),
                      "On shelf": "tinted to order" if shade.get("tinted") else f'{p["stock"]} cans'} for p in packs]
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

            if sel:
                scheme = _scheme(shade)

                def _jpeg() -> bytes:
                    base_alpha = _memo(("base_alpha", sel), lambda: result.selection_alpha(list(sel)))
                    after = _after(result, sel, ets, scheme, base_alpha, alpha)
                    buf = io.BytesIO()
                    Image.fromarray(side_by_side(result.image, after)).save(buf, format="JPEG", quality=90)
                    return buf.getvalue()

                data = _memo(("download", sel, ets, scheme), _jpeg)
                st.download_button("Save before / after", data, file_name=f"{shade['shade_name'].replace(' ', '_')}_preview.jpg",
                                   mime="image/jpeg", type="primary", use_container_width=True)
