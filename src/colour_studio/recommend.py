"""
Room-aware shade suggestions.

1. Read the room's existing colours — floor, bed, sofa, doors, curtains — from
   the segmentation (or from dominant colours if no model is available).
2. Score every wall paint we sell against them with simple, explainable colour
   rules (same family, pleasing contrast, warm neutral with wood, don't blend
   into the bed), plus the customer's words (room type, mood) and stock.
3. Return a varied top list, each with a one-line reason.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.config import SHADES_CATALOG_PATH

# Label → (friendly name, can be wooden)
FEATURE_CLASSES = {
    "floor": ("floor", True), "bed": ("bed", True), "sofa": ("sofa", False), "armchair": ("armchair", False),
    "cabinet": ("cabinet", True), "wardrobe": ("wardrobe", True), "chest of drawers": ("drawers", True),
    "door": ("door", True), "table": ("table", True), "desk": ("desk", True), "curtain": ("curtains", False),
    "rug": ("rug", False), "cushion": ("cushions", False), "pillow": ("bedding", False), "chair": ("chairs", True),
}

ROOM_WORDS = {"bedroom", "living", "kitchen", "dining", "bathroom", "kids", "study", "office", "hallway", "puja",
              "nursery", "balcony", "exterior", "facade"}
MOOD_WORDS = {"calm", "warm", "cozy", "cosy", "bold", "bright", "light", "airy", "modern", "fresh", "earthy",
              "peaceful", "elegant", "luxury", "neutral", "festive", "cheerful", "minimal", "clean", "romantic"}
MOOD_SYNONYMS = {"calm": ["peaceful", "airy"], "cozy": ["warm"], "cosy": ["warm"], "fresh": ["airy", "refreshing"],
                 "light": ["airy", "bright"], "soft": ["gentle", "calm"]}


# --- colour maths ---------------------------------------------------------------

def rgb_to_lab(rgb) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32).reshape(1, -1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    lab = np.atleast_2d(lab)
    c = np.hypot(lab[:, 1], lab[:, 2])
    h = (np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) + 360.0) % 360.0
    return np.stack([lab[:, 0], c, h], axis=1)


def delta_e(lab1, lab2) -> float:
    return float(np.linalg.norm(np.asarray(lab1, dtype=np.float32) - np.asarray(lab2, dtype=np.float32)))


def hue_gap(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def hex_to_rgb(hex_code: str):
    h = hex_code.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def is_wood(lch) -> bool:
    L, C, H = lch
    return 18 <= L <= 78 and C >= 10 and 25 <= H <= 85


def colour_word(lch) -> str:
    """A plain colour name for a room colour."""
    L, C, H = lch
    if C < 8:
        return "white" if L > 85 else ("light grey" if L > 65 else ("grey" if L > 35 else "black"))
    if 25 <= H < 85:
        if L > 75:
            return "cream"
        return "wood brown" if C < 45 else "orange"
    if H < 25 or H >= 345:
        return "red" if C > 35 else "pink"
    if 85 <= H < 110:
        return "yellow"
    if 110 <= H < 170:
        return "green"
    if 170 <= H < 230:
        return "teal"
    if 230 <= H < 300:
        return "blue"
    return "purple"


# --- read the room ---------------------------------------------------------------

def room_palette(image_rgb: np.ndarray, label_map: Optional[np.ndarray], id2label: Dict[int, str],
                 wall_mask: Optional[np.ndarray] = None, max_items: int = 5) -> List[Dict[str, Any]]:
    """
    The room's main non-wall colours, each with a friendly name like
    'wooden floor' or 'white bedding'.
    """
    total = image_rgb.shape[0] * image_rgb.shape[1]
    items: List[Dict[str, Any]] = []
    if label_map is not None and id2label:
        name_to_id = {v: k for k, v in id2label.items()}
        for cls, (friendly, woodish) in FEATURE_CLASSES.items():
            cid = name_to_id.get(cls)
            if cid is None:
                continue
            m = (label_map == cid)
            if wall_mask is not None:
                m &= ~wall_mask
            m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            share = m.sum() / total
            if share < 0.01:
                continue
            rgb = np.median(image_rgb[m], axis=0)
            lch = lab_to_lch(rgb_to_lab(rgb))[0]
            if woodish and is_wood(lch):
                label = f"wooden {friendly}"
            else:
                label = f"{colour_word(lch)} {friendly}"
            items.append({"name": label, "feature": friendly, "rgb": tuple(int(v) for v in rgb),
                          "lab": rgb_to_lab(rgb)[0], "lch": lch, "share": float(share),
                          "is_big_furniture": friendly in ("bed", "sofa", "wardrobe")})
    if not items:  # no model: dominant colours outside the selected wall
        pix = image_rgb.reshape(-1, 3)
        if wall_mask is not None:
            pix = image_rgb[~wall_mask]
        if len(pix) > 20000:
            pix = pix[np.random.default_rng(0).choice(len(pix), 20000, replace=False)]
        if len(pix) >= 50:
            k = 5
            _, labels, centres = cv2.kmeans(pix.astype(np.float32), k, None,
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2,
                                            cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.ravel(), minlength=k)
            for idx in np.argsort(-counts):
                rgb = centres[idx]
                lch = lab_to_lch(rgb_to_lab(rgb))[0]
                word = colour_word(lch)
                items.append({"name": f"{'wooden' if is_wood(lch) else word} tones", "feature": "room",
                              "rgb": tuple(int(v) for v in rgb), "lab": rgb_to_lab(rgb)[0], "lch": lch,
                              "share": float(counts[idx] / len(labels)), "is_big_furniture": False})
    items.sort(key=lambda it: it["share"], reverse=True)
    # Drop near-duplicates (e.g. wooden floor and wooden bed of the same tone keep both names, but one colour)
    unique: List[Dict[str, Any]] = []
    for it in items:
        if all(delta_e(it["lab"], u["lab"]) > 6 or it["feature"] != u["feature"] for u in unique):
            unique.append(it)
    return unique[:max_items]


def current_wall_colour(image_rgb: np.ndarray, alpha: np.ndarray) -> Optional[Dict[str, Any]]:
    m = alpha > 0.6
    if m.sum() < 50:
        return None
    rgb = np.median(image_rgb[m], axis=0)
    return {"rgb": tuple(int(v) for v in rgb), "lab": rgb_to_lab(rgb)[0]}


# --- score shades -----------------------------------------------------------------

def _query_words(query: str) -> List[str]:
    words = [w.strip(".,!?").lower() for w in (query or "").split()]
    out = []
    for w in words:
        if w in ROOM_WORDS or w in MOOD_WORDS:
            out.append(w)
            out.extend(MOOD_SYNONYMS.get(w, []))
    return out


def score_shade(shade: Dict[str, Any], palette: List[Dict[str, Any]], wall: Optional[Dict[str, Any]],
                words: List[str]) -> Dict[str, Any]:
    lab = rgb_to_lab(hex_to_rgb(shade["hex_code"]))[0]
    L, C, H = lab_to_lch(lab)[0]
    score, reasons = 0.0, []  # reasons: (weight, text)

    for item in palette[:4]:
        aL, aC, aH = item["lch"]
        weight = 0.6 + min(item["share"], 0.3)          # bigger things matter more
        if item.get("is_big_furniture") and delta_e(lab, item["lab"]) < 14:
            score -= 0.6
            reasons.append((-1.0, f"close to the {item['name']}, so it may blend in"))
            continue
        neutral_shade, neutral_item = C < 10, aC < 10
        wood = item["name"].startswith("wooden")
        if neutral_shade:
            s = 0.35
            if wood and 50 <= H <= 110:
                s, txt = 0.7, f"warm neutral that suits the {item['name']}"
            else:
                txt = f"easy neutral with the {item['name']}"
        elif neutral_item:
            s, txt = 0.3, f"stands out against the {item['name']}"
        else:
            gap = hue_gap(H, aH)
            same_word = colour_word((L, C, H)) == colour_word(item["lch"])
            if gap < 35 and same_word:
                s, txt = 0.55, f"same colour family as the {item['name']}"
            elif gap < 35:
                s, txt = 0.4, f"a close, harmonious tone next to the {item['name']}"
            elif 140 <= gap <= 220:
                s, txt = 0.75, f"contrasts beautifully with the {item['name']}"
            elif 90 <= gap < 140:
                s, txt = 0.25, f"a lively mix with the {item['name']}"
            else:
                s, txt = 0.05, ""
        score += s * weight
        if txt:
            reasons.append((s * weight, txt))

    if wall is not None:
        d = delta_e(lab, wall["lab"])
        if d < 10:
            score += 0.25
            reasons.append((0.3, "close to the current wall colour — a safe refresh"))

    tags = [t.lower() for t in shade.get("room_tags", [])]
    hits = list(dict.fromkeys(w for w in words if any(w in t for t in tags)))
    if hits:
        score += 0.45 * len(hits)

    stock = int(shade.get("total_stock", 0))
    if stock <= 0:
        score -= 0.4
    elif stock < 15:
        score += 0.05
    else:
        score += 0.2

    reasons.sort(key=lambda r: r[0], reverse=True)
    positive = [t for w, t in reasons if w > 0]
    room_reason = positive[0] if positive else ""
    if hits:
        mood = " ".join(hits[:2])
        reason = f"{mood} pick — {room_reason}" if room_reason else f"fits a {mood} feel"
    else:
        reason = room_reason or ("popular all-round choice" if C < 20 else "a bold accent option")
    warnings = [t for w, t in reasons if w < 0]
    if warnings:
        reason += f"; {warnings[0]}"
    reason = reason[0].upper() + reason[1:]
    return {**shade, "score": round(score, 3), "reason": reason, "lab": lab}


def load_wall_paints(include_exterior: bool = False) -> List[Dict[str, Any]]:
    with open(SHADES_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    allowed = {"Interior Emulsion"} | ({"Exterior Emulsion"} if include_exterior else set())
    return [s for s in catalog if s["category"] in allowed]


def recommend_shades(palette: List[Dict[str, Any]], wall: Optional[Dict[str, Any]] = None, query: str = "",
                     limit: int = 5, include_exterior: bool = False) -> List[Dict[str, Any]]:
    """Top shades for this room, varied so the list isn't five near-identical beiges."""
    from src.colour_assistant import enrich_shades_with_stock

    words = _query_words(query)
    exterior = include_exterior or any(w in ("exterior", "facade") for w in words)
    shades = enrich_shades_with_stock(load_wall_paints(include_exterior=exterior))
    scored = sorted((score_shade(s, palette, wall, words) for s in shades), key=lambda s: s["score"], reverse=True)
    picks: List[Dict[str, Any]] = []
    for s in scored:
        if all(delta_e(s["lab"], p["lab"]) > 15 for p in picks):
            picks.append(s)
        if len(picks) == limit:
            break
    for p in picks:
        p.pop("lab", None)
    return picks


def enrich_wall_paints(include_exterior: bool = False) -> List[Dict[str, Any]]:
    """Wall paints we sell, with stock and pack options."""
    from src.colour_assistant import enrich_shades_with_stock

    return enrich_shades_with_stock(load_wall_paints(include_exterior=include_exterior))
