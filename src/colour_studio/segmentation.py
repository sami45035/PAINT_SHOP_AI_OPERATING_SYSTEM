"""
Wall finding for Colour Studio.

Main path: SegFormer (trained on the ADE20K indoor/outdoor dataset, which has a
dedicated "wall" class) running through ONNX Runtime on the CPU. It separates
walls from the ceiling, floor, doors, windows, furniture and pictures.

Fallback path (no model available): the customer taps a wall and we grow a
region from that point, stopping at strong edges.

Either way the result is split into separate walls so one wall can be painted
as an accent.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    COLOUR_WORKING_MAX_SIDE,
    MODELS_DIR,
    WALL_MODEL_FILES,
    WALL_MODEL_FLIP_PASS,
    WALL_MODEL_INPUT_SIZE,
    WALL_MODEL_MAX_LONG_SIDE,
    WALL_MODEL_PATH,
    WALL_MODEL_REPO,
)

log = logging.getLogger(__name__)

# ADE20K (150-class) ids we care about. Used only if the model's config.json
# can't be read; normally labels come from the downloaded config.
ADE20K_FALLBACK_LABELS: Dict[int, str] = {
    0: "wall", 1: "building", 3: "floor", 5: "ceiling", 7: "bed", 8: "windowpane", 10: "cabinet",
    14: "door", 15: "table", 18: "curtain", 19: "chair", 22: "painting", 23: "sofa", 24: "shelf",
    25: "house", 27: "mirror", 28: "rug", 30: "armchair", 33: "desk", 35: "wardrobe", 36: "lamp",
    39: "cushion", 44: "chest of drawers", 57: "pillow",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------

@dataclass
class WallRegion:
    id: int
    name: str
    mask: np.ndarray          # bool, working resolution
    area_share: float         # share of the whole image
    centroid: Tuple[float, float]  # (x, y) normalised


@dataclass
class SegmentationResult:
    image: np.ndarray                     # RGB uint8 at working resolution
    wall_soft: np.ndarray                 # float32 0..1, edge-refined wall probability
    regions: List[WallRegion]
    label_map: Optional[np.ndarray] = None  # int class id per pixel (model path only)
    id2label: Dict[int, str] = field(default_factory=dict)
    method: str = "model"                 # "model" or "tap"
    seconds: float = 0.0

    @property
    def wall_mask(self) -> np.ndarray:
        return self.wall_soft > 0.5

    def region_at(self, x: float, y: float) -> Optional[WallRegion]:
        """Region under a normalised (x, y) point, if any."""
        h, w = self.wall_soft.shape
        px, py = int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))
        for r in self.regions:
            if r.mask[py, px]:
                return r
        # Tapped just outside a wall edge: take the nearest region within 12 px
        best, best_d = None, 12.0
        for r in self.regions:
            ys, xs = np.nonzero(r.mask[max(0, py - 12):py + 13, max(0, px - 12):px + 13])
            if len(xs):
                d = float(np.min(np.hypot(xs + max(0, px - 12) - px, ys + max(0, py - 12) - py)))
                if d < best_d:
                    best, best_d = r, d
        return best

    def selection_alpha(self, region_ids: List[int]) -> np.ndarray:
        """Soft paint mask (0..1) for the chosen regions."""
        if not region_ids:
            return np.zeros_like(self.wall_soft)
        chosen = np.zeros(self.wall_soft.shape, dtype=bool)
        for r in self.regions:
            if r.id in region_ids:
                chosen |= r.mask
        # Grow a little so the soft edge of the wall is included, then keep the soft values
        grown = cv2.dilate(chosen.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        alpha = np.where(grown, self.wall_soft, 0.0).astype(np.float32)
        alpha[chosen] = np.maximum(alpha[chosen], 0.98)
        return alpha


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def to_working_size(image_rgb: np.ndarray, max_side: int = COLOUR_WORKING_MAX_SIDE) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    if image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    return np.ascontiguousarray(image_rgb)


def guided_filter(guide_gray: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware smoothing (He et al.). Snaps a blurry mask to real image edges."""
    I = guide_gray.astype(np.float32)
    p = src.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(I, -1, k)
    mean_p = cv2.boxFilter(p, -1, k)
    corr_Ip = cv2.boxFilter(I * p, -1, k)
    corr_II = cv2.boxFilter(I * I, -1, k)
    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def _edges(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 30, 9)
    return cv2.Canny(gray, 12, 36)


def _edge_barrier(image_rgb: np.ndarray) -> np.ndarray:
    """All clear edges (used when growing a region from a tap)."""
    return cv2.dilate(_edges(image_rgb), np.ones((3, 3), np.uint8)).astype(bool)


def _side_samples(lab: np.ndarray, x1, y1, x2, y2, offset: int, n: int = 24):
    """Lab pixels just left and right of a segment (perpendicular offset)."""
    h, w = lab.shape[:2]
    length = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length   # unit normal
    t = np.linspace(0.1, 0.9, n)
    xs, ys = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    pts = []
    for sign in (-1, 1):
        px = np.clip(np.round(xs + sign * nx * offset).astype(int), 0, w - 1)
        py = np.clip(np.round(ys + sign * ny * offset).astype(int), 0, h - 1)
        pts.append((px, py))
    return pts


def _looks_like_corner(lab, region, x1, y1, x2, y2) -> bool:
    """
    A wall corner has the SAME paint on both sides, just lit differently:
    similar colour (a, b) but a brightness step. A door or window frame has
    a different material on one side; a picture frame has the picture.
    """
    (lx, ly), (rx, ry) = _side_samples(lab, x1, y1, x2, y2, offset=8)
    (lx2, ly2), (rx2, ry2) = _side_samples(lab, x1, y1, x2, y2, offset=18)
    if region is not None:
        both = region[ly, lx] & region[ry, rx] & region[ly2, lx2] & region[ry2, rx2]
        if both.mean() < 0.7:
            return False
        lx, ly, rx, ry = lx[both], ly[both], rx[both], ry[both]
        lx2, ly2, rx2, ry2 = lx2[both], ly2[both], rx2[both], ry2[both]
    left, left2 = np.median(lab[ly, lx], axis=0), np.median(lab[ly2, lx2], axis=0)
    right, right2 = np.median(lab[ry, rx], axis=0), np.median(lab[ry2, rx2], axis=0)
    # Each side must be one even surface (a frame + its picture is not)...
    if np.linalg.norm(left - left2) > 4.0 or np.linalg.norm(right - right2) > 4.0:
        return False
    # ...with the same paint colour on both sides, only lit differently.
    chroma_gap = float(np.hypot(left[1] - right[1], left[2] - right[2]))
    step = abs(float(left[0] - right[0]))
    return chroma_gap < 5.0 and 1.5 <= step <= 30.0


def _ceiling_kinks(wall_bin: np.ndarray) -> List[int]:
    """
    Wall corners from the shape of the wall area: the line where walls meet the
    ceiling changes direction at every corner (flat on the back wall, sloping on
    the side walls in a photo). Returns x positions of those bends.

    Each candidate bend is confirmed by fitting a straight line on both sides
    (robustly, so a lamp-glow bite in part of the outline doesn't spoil it).
    """
    from src.colour_studio.refine import _fit_line

    h, w = wall_bin.shape
    has = wall_bin.any(axis=0)
    if has.sum() < w * 0.5:
        return []
    raw_top = np.where(has, wall_bin.argmax(axis=0), -1)
    valid = raw_top >= 0
    xs = np.arange(w)
    top = np.interp(xs, xs[valid], raw_top[valid]).astype(np.float32)
    if w >= 15:  # running median smooths out small bumps in the outline
        padded = np.pad(top, 7, mode="edge")
        top = np.median(np.lib.stride_tricks.sliding_window_view(padded, 15), axis=1).astype(np.float32)
    k = max(6, w // 30)
    kl = max(k, w // 12)
    d = np.zeros(w, np.float32)
    d[k:w - k] = np.abs((top[2 * k:] - top[k:w - k]) / k - (top[k:w - k] - top[:w - 2 * k]) / k)

    margin = int(0.04 * w)
    out: List[int] = []
    tried = 0
    while tried < 10:
        x = int(d.argmax())
        if d[x] < 0.2:
            break
        d[max(0, x - 2 * k):x + 2 * k] = 0
        tried += 1
        if not (margin <= x <= w - margin):
            continue
        lines = []
        for a_, b_ in ((max(0, x - kl), x), (x, min(w, x + kl + 1))):
            sel = (raw_top[a_:b_] > 3)          # ignore columns where the wall runs off the photo
            if sel.sum() < 0.6 * (b_ - a_):
                break
            xx = xs[a_:b_][sel].astype(np.float32)
            yy = raw_top[a_:b_][sel].astype(np.float32)
            fit = _fit_line(xx, yy, tol=2.0, iters=60)
            if fit is None or fit[2].sum() < 0.6 * len(xx):
                break
            lines.append(fit[:2])
        if len(lines) != 2:
            continue
        (a1, b1), (a2, b2) = lines
        meet_gap = abs((a1 * x + b1) - (a2 * x + b2))
        if abs(a2 - a1) >= 0.25 and meet_gap <= 8:
            out.append(x)
    return sorted(out)


def _corner_lines(image_rgb: np.ndarray, region: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Wall corners as a barrier. Long near-vertical lines with the same paint on
    both sides are corners and are extended top to bottom (often only part of a
    corner is crisp). Other lines (frames, furniture edges) are ignored here.
    Texture, switches and small marks don't make long lines, so they don't cut
    a wall in two.
    """
    h, w = image_rgb.shape[:2]
    edges = _edges(image_rgb)
    zone = None
    if region is not None:
        zone = cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        edges[~zone] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=int(0.08 * min(h, w)), maxLineGap=20)
    barrier = np.zeros((h, w), np.uint8)
    if lines is not None:
        lab = cv2.cvtColor(image_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            dx, dy = x2 - x1, y2 - y1
            vertical = abs(dy) > 0 and abs(dx) / abs(dy) < 0.2
            long_enough = abs(dy) >= 0.22 * h   # corners are tall; picture frames are not
            if vertical and long_enough and _looks_like_corner(lab, region, x1, y1, x2, y2):
                slope = dx / dy
                x_top = int(round(x1 - slope * y1))
                x_bot = int(round(x1 + slope * (h - 1 - y1)))
                cv2.line(barrier, (x_top, 0), (x_bot, h - 1), 1, 3)
    if zone is not None:
        barrier &= zone.astype(np.uint8)
    return barrier.astype(bool)


def _outline_slope(mask: np.ndarray, which: str = "top") -> Optional[float]:
    """Slope (dy/dx) of a wall's top or bottom outline, from its straightest part."""
    from src.colour_studio.refine import _fit_line

    h, w = mask.shape
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) < 20:
        return None
    if which == "top":
        ys = mask[:, cols].argmax(axis=0)
        ok = ys > 2                                   # not running off the top of the photo
    else:
        ys = h - 1 - mask[::-1, cols].argmax(axis=0)
        ok = ys < h - 3
    if ok.sum() < max(15, 0.4 * len(cols)):
        return None
    fit = _fit_line(cols[ok].astype(np.float32), ys[ok].astype(np.float32), tol=3.0)
    if fit is None or fit[2].sum() < 0.4 * ok.sum():
        return None
    return fit[0]


def _wall_side(r: WallRegion) -> Optional[str]:
    """
    Which wall is this, from perspective: the ceiling line is flat on the wall
    you face, slopes down toward the middle on a left wall and the other way
    on a right wall (the floor line does the opposite).
    """
    top = _outline_slope(r.mask, "top")
    if top is not None:
        if abs(top) < 0.12:
            return "Back"
        return "Left" if top > 0 else "Right"
    bottom = _outline_slope(r.mask, "bottom")
    if bottom is not None:
        if abs(bottom) < 0.12:
            return "Back"
        return "Left" if bottom < 0 else "Right"
    return None


def _name_regions(regions: List[WallRegion]) -> List[WallRegion]:
    """Give walls names a shop assistant would say: Back wall, Left wall, Right wall."""
    if not regions:
        return regions
    names: Dict[int, str] = {}
    for r in regions:
        side = _wall_side(r)
        if side is None:  # no clear outline: fall back to where it sits in the photo
            x = r.centroid[0]
            side = "Left" if x < 0.35 else ("Right" if x > 0.65 else "Back")
        names[r.id] = f"{side} wall"
    # Number duplicates from left to right: "Left wall", "Left wall 2"
    seen: Dict[str, int] = {}
    for r in sorted(regions, key=lambda r: r.centroid[0]):
        base = names[r.id]
        seen[base] = seen.get(base, 0) + 1
        r.name = base if seen[base] == 1 else f"{base} {seen[base]}"
    return sorted(regions, key=lambda r: r.centroid[0])


def split_into_walls(image_rgb: np.ndarray, wall_bin: np.ndarray, min_share: float = 0.012) -> List[WallRegion]:
    """
    Split the wall area into separate walls. Corners between walls usually show
    a crisp brightness step, so strong edges cut the wall area into pieces;
    watershed then gives every leftover wall pixel to its nearest piece.
    """
    h, w = wall_bin.shape
    total = h * w
    if wall_bin.sum() < total * min_share:
        return []
    barrier = _corner_lines(image_rgb, wall_bin)
    kinks = np.zeros((h, w), np.uint8)
    for x in _ceiling_kinks(wall_bin):
        cv2.line(kinks, (x, 0), (x, h - 1), 1, 3)
    barrier |= kinks.astype(bool) & wall_bin
    cores = (wall_bin & ~barrier).astype(np.uint8)
    cores = cv2.morphologyEx(cores, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=4)

    markers = np.zeros((h, w), dtype=np.int32)
    markers[~wall_bin] = 1  # background
    next_id = 2
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= total * min_share:
            markers[lab == i] = next_id
            next_id += 1
    if next_id == 2:  # nothing big enough — treat all wall as one region
        markers[wall_bin] = 2
        next_id = 3

    cv2.watershed(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), markers)
    # Watershed marks boundaries with -1: give them to a neighbouring wall
    boundary = markers == -1
    if boundary.any():
        grown = cv2.dilate(np.where(markers > 1, markers, 0).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        markers[boundary] = grown[boundary]

    regions = _merge_similar(image_rgb, markers, wall_bin, list(range(2, next_id)), corners=barrier)
    out = []
    for rid, m in enumerate(regions, start=1):
        share = float(m.sum()) / total
        if share < min_share * 0.5:
            continue
        ys, xs = np.nonzero(m)
        out.append(WallRegion(id=rid, name="", mask=m, area_share=share, centroid=(xs.mean() / w, ys.mean() / h)))
    return _name_regions(out)


def _merge_similar(image_rgb: np.ndarray, markers: np.ndarray, wall_bin: np.ndarray, ids: List[int],
                   max_delta_e: float = 3.0, corners: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """
    Re-join neighbouring pieces that are really one wall. Compares the colour
    right at the shared border (not the whole-piece average), so a gentle light
    gradient across one wall doesn't stop it being re-joined, while a real
    corner — a brightness step right at the border — keeps walls apart.
    """
    lab = cv2.cvtColor(image_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    masks = {i: (markers == i) & wall_bin for i in ids}
    masks = {i: m for i, m in masks.items() if m.any()}
    parent = {i: i for i in masks}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    near = np.ones((9, 9), np.uint8)
    band_k = np.ones((25, 25), np.uint8)
    corner_zone = None
    if corners is not None and corners.any():
        corner_zone = cv2.dilate(corners.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    keys = list(masks)
    grown = {i: cv2.dilate(masks[i].astype(np.uint8), near).astype(bool) for i in keys}
    for a_idx, a in enumerate(keys):
        for b in keys[a_idx + 1:]:
            border = grown[a] & grown[b]
            if border.sum() < 40:
                continue
            if corner_zone is not None and (border & corner_zone).sum() > 0.5 * border.sum():
                continue  # they meet at a detected corner: two walls, even if painted alike
            band = cv2.dilate(border.astype(np.uint8), band_k).astype(bool)
            pa, pb = lab[band & masks[a]], lab[band & masks[b]]
            if len(pa) < 30 or len(pb) < 30:
                continue
            if np.linalg.norm(np.median(pa, axis=0) - np.median(pb, axis=0)) < max_delta_e:
                parent[find(b)] = find(a)
    merged: Dict[int, np.ndarray] = {}
    for i, m in masks.items():
        root = find(i)
        merged[root] = merged.get(root, np.zeros_like(m)) | m
    return list(merged.values())


# --------------------------------------------------------------------------------
# Model path
# --------------------------------------------------------------------------------

def _model_dir() -> Path:
    return MODELS_DIR / WALL_MODEL_REPO.replace("/", "__")


def find_local_model() -> Optional[Path]:
    if WALL_MODEL_PATH and Path(WALL_MODEL_PATH).exists():
        return Path(WALL_MODEL_PATH)
    d = _model_dir()
    for f in WALL_MODEL_FILES:
        p = d / Path(f).name
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    return None


def download_model(progress: Optional[Callable[[float, str], None]] = None, timeout: int = 60) -> Tuple[Optional[Path], str]:
    """
    Download the ONNX model + config from Hugging Face into models/.
    Returns (path, message). Never raises — on failure path is None.
    """
    existing = find_local_model()
    if existing:
        return existing, "Model already downloaded."
    try:
        import requests
    except ImportError:
        return None, "The `requests` package is missing, so the model can't be downloaded."

    d = _model_dir()
    d.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{WALL_MODEL_REPO}/resolve/main/"
    try:
        r = requests.get(base + "config.json", timeout=timeout)
        if r.ok:
            (d / "config.json").write_bytes(r.content)
    except Exception as exc:  # config is optional
        log.warning("Could not fetch model config: %s", exc)

    last_err = "unknown error"
    for f in WALL_MODEL_FILES:
        target = d / Path(f).name
        tmp = target.with_suffix(".part")
        try:
            with requests.get(base + f, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    last_err = f"{f}: HTTP {resp.status_code}"
                    continue
                total = int(resp.headers.get("content-length", 0)) or None
                done = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if progress and total:
                            progress(done / total, f"Downloading wall model… {done / 1e6:.0f} / {total / 1e6:.0f} MB")
            tmp.replace(target)
            return target, f"Downloaded {target.name} ({target.stat().st_size / 1e6:.0f} MB)."
        except Exception as exc:
            last_err = f"{f}: {exc}"
            if tmp.exists():
                tmp.unlink()
    return None, f"Couldn't download the wall model ({last_err})."


def _read_id2label(model_path: Path) -> Dict[int, str]:
    cfg = model_path.parent / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            return {int(k): str(v).split(",")[0].strip().lower() for k, v in data.get("id2label", {}).items()}
        except Exception:
            pass
    return dict(ADE20K_FALLBACK_LABELS)


def _straighten_regions(regions: List[WallRegion], edges: np.ndarray, objects: np.ndarray, wall: np.ndarray):
    """Run the straight-ceiling-line fix on each wall, then refresh its stats."""
    from src.colour_studio.refine import straighten_top

    if not regions:
        return regions, wall
    h, w = wall.shape
    union = np.zeros_like(wall)
    for r in regions:
        others = np.zeros_like(wall)
        for o in regions:
            if o.id != r.id:
                others |= o.mask
        r.mask = straighten_top(r.mask, edges, objects, others)
        ys, xs = np.nonzero(r.mask)
        r.area_share = float(r.mask.mean())
        r.centroid = (xs.mean() / w, ys.mean() / h)
        union |= r.mask
    return _name_regions(regions), union


class WallSegmenter:
    """Runs SegFormer on a photo and returns walls plus a full label map."""

    def __init__(self, session, id2label: Dict[int, str], input_size: int = WALL_MODEL_INPUT_SIZE,
                 flip_pass: bool = WALL_MODEL_FLIP_PASS):
        self.session = session
        self.id2label = id2label
        self.input_size = input_size
        self.flip_pass = flip_pass
        self.square_only = False            # set if the model file only accepts square input
        self.input_name = session.get_inputs()[0].name
        names = {v: k for k, v in id2label.items()}
        self.wall_ids = [names.get("wall", 0)]
        self.exterior_ids = [i for i in (names.get("building"), names.get("house")) if i is not None]

    @classmethod
    def from_path(cls, model_path: Path) -> "WallSegmenter":
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        return cls(session, _read_id2label(model_path))

    def _input_size(self, h: int, w: int) -> Tuple[int, int]:
        """Keep the photo's shape: short side = input_size, both sides multiples of 32."""
        if self.square_only:
            return self.input_size, self.input_size
        s = self.input_size / min(h, w)
        ih, iw = h * s, w * s
        cap = WALL_MODEL_MAX_LONG_SIDE / max(ih, iw)
        if cap < 1:
            ih, iw = ih * cap, iw * cap
        return max(32, int(round(ih / 32)) * 32), max(32, int(round(iw / 32)) * 32)

    def _run(self, image_rgb: np.ndarray) -> np.ndarray:
        ih, iw = self._input_size(*image_rgb.shape[:2])
        x = cv2.resize(image_rgb, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None].astype(np.float32))
        try:
            out = self.session.run(None, {self.input_name: x})[0]
        except Exception:
            if self.square_only:
                raise
            self.square_only = True          # older exports: fixed 512 x 512 input
            return self._run(image_rgb)
        return out[0]                         # (C, h/4, w/4)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=0, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=0, keepdims=True)

    def _probs(self, image_rgb: np.ndarray) -> np.ndarray:
        p = self._softmax(self._run(image_rgb))
        if self.flip_pass:
            q = self._softmax(self._run(np.ascontiguousarray(image_rgb[:, ::-1])))[:, :, ::-1]
            if q.shape == p.shape:
                p = (p + q) / 2.0
        return p

    def _logits(self, image_rgb: np.ndarray) -> np.ndarray:  # kept for older callers
        return self._run(image_rgb)

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        import time

        from src.colour_studio.refine import feather, refine_wall

        t0 = time.perf_counter()
        img = to_working_size(image_rgb)
        H, W = img.shape[:2]
        probs = self._probs(img)                        # (C, h, w) at 1/4 of the model input
        C = probs.shape[0]

        wall_ids = list(self.wall_ids)
        labels_low = probs.argmax(axis=0)
        share = lambda ids: float(np.isin(labels_low, ids).mean())
        if share(wall_ids) < 0.03 and self.exterior_ids and share(self.exterior_ids) > 0.10:
            wall_ids = wall_ids + self.exterior_ids     # outside photo: paint the building facade

        p_wall = cv2.resize(probs[wall_ids].sum(axis=0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        conf = cv2.resize(probs.max(axis=0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        label_map = cv2.resize(labels_low.astype(np.uint8 if C <= 255 else np.uint16), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(np.int32)

        wall, edges, objects = refine_wall(img, p_wall, label_map, conf, self.id2label, return_parts=True)
        regions = split_into_walls(img, wall)
        regions, wall = _straighten_regions(regions, edges, objects, wall)
        return SegmentationResult(
            image=img, wall_soft=feather(wall, img), regions=regions, label_map=label_map,
            id2label=self.id2label, method="model", seconds=time.perf_counter() - t0,
        )


# --------------------------------------------------------------------------------
# Fallback: tap-to-select without a model
# --------------------------------------------------------------------------------

def region_from_tap(image_rgb: np.ndarray, x: float, y: float, tolerance: float = 9.0) -> np.ndarray:
    """
    Grow a wall region from a tapped point (normalised x, y). Stops at strong
    edges and where colour changes more than `tolerance` (Lab units) from a
    neighbour. Returns a bool mask at the image's resolution.
    """
    h, w = image_rgb.shape[:2]
    px, py = int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))
    smooth = cv2.bilateralFilter(image_rgb, 9, 30, 9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    barrier = _edge_barrier(image_rgb) | _corner_lines(image_rgb)
    # Blown-out / near-white glare (lamp hotspots, a white ceiling in glare) has no
    # usable colour: a flood through it can bridge from a wall onto the ceiling.
    blown = (image_rgb.min(axis=2) >= 240) & ((image_rgb.max(axis=2).astype(np.int16) - image_rgb.min(axis=2)) < 14)
    if blown[py, px]:   # tapped a hotspot: start from the nearest pixel that has colour
        win = 60
        y0, x0 = max(0, py - win), max(0, px - win)
        cand = np.argwhere(~blown[y0:py + win + 1, x0:px + win + 1])
        if len(cand):
            d = (cand[:, 0] + y0 - py) ** 2 + (cand[:, 1] + x0 - px) ** 2
            py, px = int(cand[d.argmin(), 0] + y0), int(cand[d.argmin(), 1] + x0)
    barrier = barrier | blown
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_mask[1:-1, 1:-1][barrier] = 1     # floodFill never crosses non-zero mask pixels
    if barrier[py, px]:                      # tapped right on an edge — nudge to a nearby flat pixel
        flat = np.argwhere(~barrier[max(0, py - 8):py + 9, max(0, px - 8):px + 9])
        if len(flat):
            py, px = flat[0][0] + max(0, py - 8), flat[0][1] + max(0, px - 8)
    tol = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(lab, flood_mask, (px, py), (0, 0, 0), tol, tol, flags)
    region = flood_mask[1:-1, 1:-1] == 255
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    # Blown-out spots fully inside the wall (a hotspot in the middle) belong to it
    inv = (~region).astype(np.uint8)
    n_h, hole_ids, hstats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n_h):
        hx, hy, hw, hh, area = hstats[i]
        if hx == 0 or hy == 0 or hx + hw >= w or hy + hh >= h or area > 0.03 * h * w:
            continue
        hole = hole_ids[hy:hy + hh, hx:hx + hw] == i
        if blown[hy:hy + hh, hx:hx + hw][hole].mean() > 0.5:
            region[hy:hy + hh, hx:hx + hw][hole] = True
    # If the fill leaked round a faint corner, cut at the corner and keep the tapped side
    cut = np.zeros((h, w), np.uint8)
    for kx in _ceiling_kinks(region):
        cv2.line(cut, (kx, 0), (kx, h - 1), 1, 3)
    if cut.any():
        n, lab_ids = cv2.connectedComponents((region & ~cut.astype(bool)).astype(np.uint8), connectivity=4)
        keep = lab_ids[py, px]
        if keep == 0:  # tap landed on the cut line itself
            near = lab_ids[max(0, py - 4):py + 5, max(0, px - 4):px + 5]
            keep = int(near.max())
        if keep:
            region = cv2.dilate((lab_ids == keep).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & region
    return region


def tap_result(image_rgb: np.ndarray, taps: List[Tuple[float, float]]) -> SegmentationResult:
    """Build a SegmentationResult from one or more taps (fallback mode)."""
    img = to_working_size(image_rgb)
    h, w = img.shape[:2]
    regions: List[WallRegion] = []
    union = np.zeros((h, w), dtype=bool)
    for i, (x, y) in enumerate(taps, start=1):
        if union[int(np.clip(y * h, 0, h - 1)), int(np.clip(x * w, 0, w - 1))]:
            continue  # this wall is already picked
        # Start strict: a smooth light gradient still passes (neighbour-to-neighbour
        # changes are tiny) but a faint corner between two walls doesn't. Loosen
        # only if that gives too small a wall (rough or noisy paint).
        m, fallback = None, None
        for tol in (2.5, 4.0, 6.0, 9.0):
            cand = region_from_tap(img, x, y, tolerance=tol) & ~union
            share = cand.mean()
            if share <= 0.4:
                fallback = cand
            if 0.03 <= share <= 0.4:
                m = cand
                break
        if m is None:
            m = fallback if fallback is not None else cand
        if m.sum() < 50:
            continue
        union |= m
        ys, xs = np.nonzero(m)
        regions.append(WallRegion(id=i, name="", mask=m, area_share=float(m.mean()),
                                  centroid=(xs.mean() / w, ys.mean() / h)))
    soft = cv2.GaussianBlur(union.astype(np.float32), (0, 0), 1.2)
    regions = _name_regions(regions)
    return SegmentationResult(image=img, wall_soft=soft, regions=regions, method="tap")


# --------------------------------------------------------------------------------
# Manual fixes: "add a missed spot" / "remove a spot"
# --------------------------------------------------------------------------------

def spot_from_tap(image_rgb: np.ndarray, x: float, y: float, max_share: float = 0.04) -> np.ndarray:
    """
    The patch under a tap, grown from the point and stopping at edges. A spot
    is a patch, not a whole wall: if the growth spreads too far (a faint edge
    let it through), it is retried with stricter colour matching, and as a
    last resort a small round brush is used.
    """
    h, w = image_rgb.shape[:2]
    limit = max_share * h * w
    for tol in (7.0, 4.0, 2.5, 1.5):
        m = region_from_tap(image_rgb, x, y, tolerance=tol)
        if 10 <= m.sum() <= limit:
            return m
    yy, xx = np.ogrid[:h, :w]
    r = 0.03 * float(np.hypot(h, w))
    return (xx - x * w) ** 2 + (yy - y * h) ** 2 <= r * r


def apply_fixes(result: SegmentationResult, fixes: List[Tuple[str, float, float]],
                selected: Optional[List[int]] = None) -> SegmentationResult:
    """
    Apply the customer's manual fixes on top of the automatic walls.
    fixes: [("add" | "remove", x, y)] with x, y normalised. Returns a new result.
    """
    import copy

    from src.colour_studio.refine import feather

    if not fixes:
        return result
    out = copy.copy(result)
    out.regions = [copy.copy(r) for r in result.regions]
    for r in out.regions:
        r.mask = r.mask.copy()
    h, w = result.wall_soft.shape
    for op, x, y in fixes:
        spot = spot_from_tap(result.image, x, y)
        if spot.sum() < 10:
            continue
        if op == "remove":
            for r in out.regions:
                r.mask &= ~spot
            continue
        # add: give the spot to the nearest selected wall (or nearest wall at all)
        px, py = int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))
        pool = [r for r in out.regions if not selected or r.id in selected] or out.regions
        if not pool:
            new_id = 1 + max([r.id for r in out.regions], default=0)
            out.regions.append(WallRegion(id=new_id, name="Wall", mask=spot.copy(), area_share=0.0, centroid=(x, y)))
            continue
        best, best_d = pool[0], float("inf")
        for r in pool:
            dist = cv2.distanceTransform((~r.mask).astype(np.uint8), cv2.DIST_L2, 3)
            d = float(dist[py, px])
            if d < best_d:
                best, best_d = r, d
        others = np.zeros((h, w), bool)
        for r in out.regions:
            if r.id != best.id:
                others |= r.mask
        best.mask |= spot & ~others
    union = np.zeros((h, w), bool)
    kept = []
    for r in out.regions:
        if r.mask.sum() < 50:
            continue
        ys, xs = np.nonzero(r.mask)
        r.area_share = float(r.mask.mean())
        r.centroid = (xs.mean() / w, ys.mean() / h)
        union |= r.mask
        kept.append(r)
    out.regions = kept
    out.wall_soft = feather(union, result.image)
    return out
