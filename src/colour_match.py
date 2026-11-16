"""
Shade Match — "what exact colour is this?" for a customer who brings a sample.

A customer walks in with a chip of old paint, a cushion, a photo from Instagram…
The counter takes a photo, taps the colour, and gets:

* the measured colour: hex, RGB, light reflectance (LRV) and a plain name
  ("light greyish blue"), plus the nearest standard colour name;
* the closest shades the shop sells (catalogue + any brand fan decks the shop
  loaded), each with a CIEDE2000 difference and a verdict a customer understands;
* which tinting base it is made from and how many of those bases are in stock —
  or, when nothing is close, a "tint to order" ticket on the right base.

Phone photos are not a spectrophotometer: the light in the room shifts the
colour. Tapping a piece of plain white paper in the same photo fixes most of
that (white balance), and the screen says so.
"""
from __future__ import annotations

import csv
import io
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import DB_PATH

PAPER_WHITE = 0.90          # plain office paper reflects about 90 % of light
SAMPLE_SHARE = 0.025        # sample a circle 2.5 % of the photo's short side
TRIM = (20, 80)             # fallback: the middle of the brightness range
HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

# Verdicts on CIEDE2000 (ΔE00): 1 ≈ the smallest difference a trained eye sees.
VERDICTS = [
    (2.0, "exact", "Exact match"),
    (5.0, "close", "Very close"),
    (10.0, "family", "Same colour family"),
]
NO_MATCH = ("none", "Not close — tint to order")


# ------------------------------------------------------------------ colour maths
def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    m = HEX_RE.match(hex_code.strip())
    if not m:
        raise ValueError(f"not a hex colour: {hex_code!r}")
    h = m.group(1)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(round(min(max(v, 0), 255))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])          # D65


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    """Linear sRGB (…, 3) → CIE Lab (…, 3), D65."""
    xyz = np.asarray(lin, dtype=float) @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def rgb_to_lab(rgb: Sequence[float]) -> np.ndarray:
    return linear_to_lab(srgb_to_linear(np.asarray(rgb, dtype=float)))


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005). Broadcasts over (…, 3)."""
    lab1, lab2 = np.asarray(lab1, dtype=float), np.asarray(lab2, dtype=float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lpm = (L1 + L2) / 2
    Cpm = (C1p + C2p) / 2
    hsum = h1p + h2p
    hpm = np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2)
    hpm = np.where(C1p * C2p == 0, hsum, hpm)
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    d_theta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def lrv(rgb: Sequence[float]) -> float:
    """Light reflectance value, 0 (black) … 100 (white): how much light the colour gives back."""
    lin = srgb_to_linear(np.asarray(rgb, dtype=float))
    return float(100 * (lin @ _M[1]))


def base_number(rgb: Sequence[float]) -> int:
    """Which tinting base a colour is made from (same rule as the shop's catalogue): 1 pastel … 4 deep."""
    r, g, b = (np.asarray(rgb, dtype=float) / 255)
    light = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 1 if light >= 0.78 else 2 if light >= 0.58 else 3 if light >= 0.36 else 4


def verdict(de: float) -> Tuple[str, str]:
    for limit, key, words in VERDICTS:
        if de <= limit:
            return key, words
    return NO_MATCH


# ------------------------------------------------------------------ names
def describe(rgb: Sequence[float]) -> str:
    """A plain name a customer understands: 'light greyish blue', 'deep terracotta', 'off-white'."""
    L, a, b = rgb_to_lab(rgb)
    C = float(np.hypot(a, b))
    h = float(np.degrees(np.arctan2(b, a)) % 360)
    if C < 5:
        return ("white" if L > 95 else "off-white" if L > 88 else "light grey" if L > 70 else "grey" if L > 45
                else "charcoal" if L > 20 else "black")
    if 50 <= h < 115 and C < 32 and L > 68:
        return "cream" if L > 88 or (h >= 85 and L > 80) else "sand" if h >= 85 else "beige"
    if 35 <= h < 90 and ((L < 45 and C < 52) or (L < 58 and C < 45)):
        hue = "brown"
    elif 33 <= h < 55 and 40 <= C < 75 and 40 <= L < 65:
        hue = "terracotta"
    elif h < 45 or h >= 350:
        hue = "pink" if L > 70 and C < 50 else "red"
    elif h < 75:
        hue = "peach" if L > 78 else "orange"
    elif h < 105:
        hue = "yellow" if C > 40 else "ochre"
    elif h < 135:
        hue = "olive" if L < 55 else "lime" if C > 45 else "yellow-green"
    elif h < 175:
        hue = "green"
    elif h < 215:
        hue = "teal"
    elif h < 245:
        hue = "sky blue" if L > 65 else "blue"
    elif h < 295:
        hue = "navy" if L < 30 else "blue"
    elif h < 325:
        hue = "lavender" if L > 70 else "purple"
    else:
        hue = "pink" if L > 60 else "magenta"
    tone = "greyish " if C < 15 else "muted " if C < 28 else ""
    light = ("pale " if L > 88 else "light " if L > 72 else "" if L > 45 else "deep " if L > 25 else "very deep ")
    if hue in ("navy", "brown", "olive", "peach", "lavender", "sky blue", "pink") or (C > 60 and L > 45):
        light = "" if hue != "brown" or L > 25 else "dark "
    return f"{light}{tone}{hue}".strip()


_WORDS = set("""alice blue antique white aqua marine azure beige bisque black blanched almond violet brown burly wood
cadet chartreuse chocolate coral cornflower silk crimson cyan dark golden rod gray grey green khaki magenta olive orange
orchid red salmon sea slate turquoise deep pink sky dim dodger fire brick floral forest fuchsia gainsboro ghost gold
honeydew hot indian indigo ivory lavender blush lawn chiffon lemon light yellow lime linen maroon medium purple spring
midnight mint cream misty rose moccasin navajo navy old lace drab pale papaya whip peach puff peru plum powder rebecca
rosy royal saddle sandy shell sienna silver snow steel tan teal thistle tomato wheat smoke""".split())


def _title(name: str) -> str:
    """'darkslateblue' → 'Dark Slate Blue' (CSS names are run together)."""
    best = {0: []}
    for i in range(1, len(name) + 1):
        for j in range(max(0, i - 11), i):
            if j in best and name[j:i] in _WORDS and (i not in best or len(best[j]) + 1 < len(best[i])):
                best[i] = best[j] + [name[j:i]]
    return " ".join(best[len(name)]).title() if len(name) in best else name.title()


def _named_colours() -> List[Tuple[str, Tuple[int, int, int]]]:
    from PIL import ImageColor

    seen, out = set(), []
    for name, value in sorted(ImageColor.colormap.items()):
        key = name.replace("grey", "gray")
        if key in seen:
            continue
        seen.add(key)
        out.append((_title(key), tuple(ImageColor.getrgb(value)[:3])))
    return out


NAMED = _named_colours()
_NAMED_LAB = rgb_to_lab(np.array([c for _, c in NAMED], dtype=float))


def nearest_name(rgb: Sequence[float]) -> Tuple[str, float]:
    """Closest standard (CSS/X11) colour name and its ΔE00."""
    d = delta_e2000(_NAMED_LAB, rgb_to_lab(rgb))
    i = int(np.argmin(d))
    return NAMED[i][0], float(d[i])


# ------------------------------------------------------------------ sampling a photo
@dataclass
class Sample:
    rgb: Tuple[float, float, float]           # what the camera saw (sRGB 0–255)
    spread: float                             # median ΔE00 of the kept pixels to their mean (texture/pattern)
    glare: float                              # share of near-white clipped pixels
    pixels: int
    corrected: bool = False
    white_rgb: Optional[Tuple[float, float, float]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def hex(self) -> str:
        return rgb_to_hex(self.rgb)


def _patch(image: np.ndarray, fx: float, fy: float, share: float = SAMPLE_SHARE) -> np.ndarray:
    h, w = image.shape[:2]
    r = max(4, int(round(share * min(h, w))))
    cx, cy = int(round(fx * (w - 1))), int(round(fy * (h - 1)))
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return image[y0:y1, x0:x1][inside].reshape(-1, 3).astype(float)


def _robust_linear(pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean linear colour of the patch's main surface: glare, shadow and specks are left out.

    Pixels far from the median brightness (more than ~3 robust standard deviations) are dropped,
    so a shiny spot covering even a third of the patch doesn't move the reading."""
    lin = srgb_to_linear(pixels)
    y = lin @ _M[1]
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    keep = lin[np.abs(y - med) <= max(3 * mad, 0.03 * med, 1e-4)]
    if len(keep) < max(3, len(lin) // 5):
        lo, hi = np.percentile(y, TRIM)
        keep = lin[(y >= lo) & (y <= hi)] if hi > lo else lin
    return keep.mean(axis=0), keep


def sample_colour(image: np.ndarray, fx: float, fy: float, white: Optional[Tuple[float, float]] = None) -> Sample:
    """The colour at (fx, fy) — fractions of width/height — ignoring glare and shadow.

    With ``white`` (the spot where plain white paper is), the photo's lighting is corrected first."""
    px = _patch(image, fx, fy)
    glare = float(np.mean(px.max(axis=1) >= 250)) if len(px) else 0.0
    mean_lin, kept = _robust_linear(px)
    warnings: List[str] = []
    corrected, white_rgb = False, None
    if white is not None:
        w_lin, _ = _robust_linear(_patch(image, white[0], white[1]))
        white_rgb = tuple(float(v) for v in linear_to_srgb(w_lin))
        w_lab = linear_to_lab(w_lin)
        if float(w_lin @ _M[1]) < 0.12:
            warnings.append("The white spot is very dark — tap the plain white paper, not a shadow.")
        elif float(np.hypot(w_lab[1], w_lab[2])) > 40:
            warnings.append("The white spot looks strongly coloured — is it really white paper?")
        else:
            gains = np.clip(PAPER_WHITE / np.maximum(w_lin, 1e-4), 0.3, 4.0)
            mean_lin = mean_lin * gains
            kept = kept * gains
            corrected = True
    rgb = tuple(float(v) for v in linear_to_srgb(mean_lin))
    labs = linear_to_lab(np.clip(kept[:: max(1, len(kept) // 3000)], 0, 1))
    spread = float(np.median(delta_e2000(labs, linear_to_lab(np.clip(mean_lin, 0, 1))))) if len(labs) else 0.0
    if spread > 6:
        warnings.append("This spot has more than one colour (pattern or texture) — tap a flat, even area.")
    if glare > 0.25:
        warnings.append("Shine or glare on the sample — tilt it away from the light and take the photo again.")
    if np.any(mean_lin > 1.0):
        warnings.append("The sample is brighter than the white paper — check you tapped the right spots.")
    return Sample(rgb=rgb, spread=spread, glare=glare, pixels=len(px), corrected=corrected,
                  white_rgb=white_rgb, warnings=warnings)


# ------------------------------------------------------------------ the shop's shades
def ensure_library(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS shade_library (
        id INTEGER PRIMARY KEY AUTOINCREMENT, brand TEXT NOT NULL, code TEXT, name TEXT NOT NULL,
        hex TEXT NOT NULL, collection TEXT, UNIQUE (brand, code, name))""")


def shop_shades(db_path=DB_PATH) -> List[Dict]:
    """Every shade the shop can sell (tinted or ready-mix) plus brand fan-deck shades the shop loaded."""
    with sqlite3.connect(db_path) as conn:
        ensure_library(conn)
        rows = conn.execute("""
            SELECT p.brand, p.sub_brand, p.shade_name, p.shade_code, p.hex_code, p.category,
                   b.shade_code AS base_code,
                   SUM((SELECT COALESCE(SUM(i.current_stock), 0) FROM inventory i
                        WHERE i.product_id = COALESCE(p.base_id, p.id))) AS stock
            FROM products p LEFT JOIN products b ON b.id = p.base_id
            WHERE COALESCE(p.is_base, 0) = 0 AND p.category NOT LIKE '%Primer%' AND p.category NOT LIKE '%Water%'
            GROUP BY p.brand, p.shade_name""").fetchall()
        lib = conn.execute("SELECT brand, code, name, hex, collection FROM shade_library").fetchall()
    out = [dict(source="shop", brand=r[0], line=r[1], name=r[2], code=r[3], hex=r[4].upper(), category=r[5],
                base_code=r[6], stock=int(r[7] or 0)) for r in rows if HEX_RE.match(r[4] or "")]
    out += [dict(source="fan deck", brand=r[0], line=r[4] or "", name=r[2], code=r[1] or "", hex=r[3].upper(),
                 category="", base_code=None, stock=None) for r in lib if HEX_RE.match(r[3] or "")]
    return out


def match(rgb: Sequence[float], limit: int = 5, shades: Optional[List[Dict]] = None) -> List[Dict]:
    """Closest shades, best first, each with ``de`` (ΔE00), ``verdict`` and ``verdict_key``."""
    shades = shop_shades() if shades is None else shades
    if not shades:
        return []
    labs = rgb_to_lab(np.array([hex_to_rgb(s["hex"]) for s in shades], dtype=float))
    d = delta_e2000(labs, rgb_to_lab(rgb))
    out = []
    for i in np.argsort(d)[:limit]:
        s = dict(shades[int(i)])
        s["de"] = float(d[int(i)])
        s["verdict_key"], s["verdict"] = verdict(s["de"])
        out.append(s)
    return out


def bases_in_stock(number: int, db_path=DB_PATH) -> List[Dict]:
    """Tinting bases with this number (AB3, RB3, …) and how many cans of each are in stock."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT p.shade_code, p.sub_brand, p.brand, COALESCE(SUM(i.current_stock), 0)
            FROM products p LEFT JOIN inventory i ON i.product_id = p.id
            WHERE p.is_base = 1 GROUP BY p.shade_code ORDER BY 4 DESC""").fetchall()
    return [dict(code=r[0], line=r[1], brand=r[2], cans=int(r[3])) for r in rows
            if r[0] and r[0][-1:].isdigit() and int(r[0][-1]) == number]


# ------------------------------------------------------------------ brand fan decks
COLUMNS = {"brand": ("brand", "company", "make"), "code": ("code", "shade_code", "shade code", "number"),
           "name": ("name", "shade_name", "shade name", "shade", "colour", "color"),
           "hex": ("hex", "hex_code", "hex code", "colour code", "color code"),
           "collection": ("collection", "line", "range", "sub_brand", "product")}


def parse_fan_deck(raw: bytes, default_brand: str = "") -> Tuple[List[Dict], List[str]]:
    """Rows from a fan-deck CSV (brand, code, name, hex[, collection]); returns (rows, problems)."""
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    heads = {h: (h or "").strip().lower() for h in (reader.fieldnames or [])}
    pick = {k: next((h for h, low in heads.items() if low in names), None) for k, names in COLUMNS.items()}
    problems: List[str] = []
    if not pick["name"] or not pick["hex"]:
        return [], ["The file needs at least a 'name' and a 'hex' column (e.g. Royale Blue, #1A365D)."]
    rows = []
    for n, r in enumerate(reader, start=2):
        hex_code = (r.get(pick["hex"]) or "").strip()
        name = (r.get(pick["name"]) or "").strip()
        m = HEX_RE.match(hex_code)
        if not name or not m:
            problems.append(f"Line {n}: skipped (needs a name and a 6-digit hex like #1A365D)")
            continue
        rows.append(dict(brand=((r.get(pick["brand"]) if pick["brand"] else "") or default_brand or "Other").strip(),
                         code=((r.get(pick["code"]) if pick["code"] else "") or "").strip(), name=name,
                         hex="#" + m.group(1).upper(),
                         collection=((r.get(pick["collection"]) if pick["collection"] else "") or "").strip()))
    return rows, problems


def save_fan_deck(rows: List[Dict], db_path=DB_PATH) -> int:
    with sqlite3.connect(db_path) as conn:
        ensure_library(conn)
        before = conn.total_changes
        conn.executemany("""INSERT INTO shade_library (brand, code, name, hex, collection) VALUES (?,?,?,?,?)
            ON CONFLICT (brand, code, name) DO UPDATE SET hex = excluded.hex, collection = excluded.collection""",
                         [(r["brand"], r["code"], r["name"], r["hex"], r["collection"]) for r in rows])
        return conn.total_changes - before


def library_summary(db_path=DB_PATH) -> List[Tuple[str, int]]:
    with sqlite3.connect(db_path) as conn:
        ensure_library(conn)
        return conn.execute("SELECT brand, COUNT(*) FROM shade_library GROUP BY brand ORDER BY 2 DESC").fetchall()


def clear_fan_decks(db_path=DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        ensure_library(conn)
        conn.execute("DELETE FROM shade_library")


FAN_DECK_TEMPLATE = "brand,code,name,hex,collection\nAsian Paints,7241,Royale Blue,#1A365D,Royale\n"
