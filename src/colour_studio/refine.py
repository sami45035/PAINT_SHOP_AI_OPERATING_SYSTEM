"""
Turns the wall model's rough, low-resolution wall map into a clean one that
follows the real edges in the photo.

The model predicts on a coarse grid (about one prediction per 8 photo pixels
on a 960 px photo), so on its own it gives:
- holes and bites: bright, lamp-lit or sun-lit wall labelled "ceiling"
- leaks: wall label spilling onto the top of a headboard or sofa
- ragged edges: blotchy borders along corners and curtains

Steps (all edge-aware, using the photo itself):
1. Doubts: gaps with no visible outline and the wall's colour are wall (a
           real frame or curtain has an outline in the photo).
2. Leaks:  cut the wall map along strong edges; small pieces that blend into
           an object (the top of a headboard) or look nothing like the wall
           are removed.
3. Snap:   every border pixel is re-decided so the border sits on the
           strongest edge nearby (no stair-steps from the coarse grid).
4. Growth: the wall grows into neighbouring pixels across smooth colour
           changes (a light pool fading out is smooth; a ceiling corner or a
           headboard edge is not). Things the model is sure are objects
           (lamps, frames, windows…) are never grown into.
5. Holes:  small enclosed gaps that aren't objects are filled.
6. Edges:  a light feather so the paint edge is crisp but not jagged.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

# Labels that are part of the room's shell rather than objects in front of it.
SHELL_LABELS = {"wall", "ceiling", "floor", "building", "house", "skyscraper", "sky", "column", "pillar"}


def lab_image(image_rgb: np.ndarray) -> np.ndarray:
    """Float Lab (L 0..100), lightly smoothed so wall texture and JPEG noise don't count as edges."""
    smooth = cv2.bilateralFilter(image_rgb, 7, 25, 7)
    return cv2.cvtColor(smooth.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def strong_edges(image_rgb: np.ndarray, low: int = 20, high: int = 55) -> np.ndarray:
    gray = cv2.bilateralFilter(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 9, 30, 9)
    return cv2.Canny(gray, low, high) > 0


def object_mask(label_map: Optional[np.ndarray], label_conf: Optional[np.ndarray], id2label: Dict[int, str],
                min_conf: float = 0.6) -> np.ndarray:
    """Pixels the model is fairly sure are an object (lamp, frame, window, furniture…)."""
    if label_map is None or not id2label:
        return np.zeros((1, 1), bool)
    shell = [i for i, n in id2label.items() if n in SHELL_LABELS]
    obj = ~np.isin(label_map, shell)
    if label_conf is not None:
        obj &= label_conf >= min_conf
    return obj


def remove_leaks(mask: np.ndarray, prob: np.ndarray, lab: np.ndarray, edges: np.ndarray,
                 objects: Optional[np.ndarray] = None, max_piece_share: float = 0.03, colour_gap: float = 12.0,
                 sure: float = 0.85) -> np.ndarray:
    """
    Drop small pieces of 'wall' that are really part of something else:
    - cut off from the main wall by an edge but blending straight into an
      object (the top of a headboard or sofa), or
    - cut off by an edge, unsure, and a clearly different colour.
    """
    total = mask.size
    edge_zone = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if objects is None or objects.shape != mask.shape:
        objects = np.zeros_like(mask)
    cut = mask & ~edge_zone
    n, ids, stats, _ = cv2.connectedComponentsWithStats(cut.astype(np.uint8), connectivity=4)
    if n <= 2:
        return mask
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max_piece_share * total]
    if not big:
        return mask
    big_mask = np.isin(ids, big)
    drop = np.zeros_like(mask)
    ring_k = np.ones((7, 7), np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max_piece_share * total or area < 20:
            continue
        piece = ids == i
        x, y, w, h = stats[i, :4]
        pad = 12
        y0, y1, x0, x1 = max(0, y - pad), min(mask.shape[0], y + h + pad), max(0, x - pad), min(mask.shape[1], x + w + pad)
        p = piece[y0:y1, x0:x1]
        ring = cv2.dilate(p.astype(np.uint8), ring_k).astype(bool) & ~p
        ez = edge_zone[y0:y1, x0:x1]
        n_ring = max(int(ring.sum()), 1)
        into_object = (ring & objects[y0:y1, x0:x1] & ~ez).sum() / n_ring
        into_wall = (ring & big_mask[y0:y1, x0:x1] & ~ez).sum() / n_ring
        if into_object > 0.15 and into_wall < 0.05:
            drop |= piece          # merges seamlessly into an object, walled off from the wall by an edge
            continue
        if float(prob[piece].mean()) >= sure:
            continue  # the model is confident: a real (small) bit of wall
        nb = big_mask[y0:y1, x0:x1] & ring
        if nb.sum() < 5:
            continue  # not touching the main wall: leave it to the hole/size checks
        here = np.median(lab[y0:y1, x0:x1][p], axis=0)
        there = np.median(lab[y0:y1, x0:x1][cv2.dilate(nb.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
                                             & big_mask[y0:y1, x0:x1]], axis=0)
        gap = float(np.hypot(here[1] - there[1], here[2] - there[2]) + 0.5 * abs(here[0] - there[0]))
        if gap > colour_gap:
            drop |= piece
    if not drop.any():
        return mask
    out = mask & ~drop
    # Edge pixels that belonged to a dropped piece go too
    edge_px = mask & ~cut
    near_drop = cv2.dilate(drop.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    near_keep = cv2.dilate((cut & ~drop).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out &= ~(edge_px & near_drop & ~near_keep)
    return out


def grow_smooth(mask: np.ndarray, lab: np.ndarray, blocked: np.ndarray, step_tol: float = 2.6,
                max_steps: int = 80, min_prob: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Grow the wall one pixel ring at a time into neighbours whose colour is close
    to the wall pixels right next to them. A smooth fade (light pool, shadow)
    passes; an edge (ceiling corner, frame, furniture) stops it.
    """
    m = mask.astype(np.float32)
    allowed = ~blocked
    if min_prob is not None:
        allowed &= min_prob
    k = np.ones((3, 3), np.float32)
    for _ in range(max_steps):
        ring = cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0
        ring &= (m == 0) & allowed
        if not ring.any():
            break
        cnt = cv2.filter2D(m, -1, k, borderType=cv2.BORDER_CONSTANT)
        mean = np.stack([cv2.filter2D(lab[..., c] * m, -1, k, borderType=cv2.BORDER_CONSTANT) for c in range(3)], -1)
        mean /= np.maximum(cnt, 1e-6)[..., None]
        diff = lab - mean
        # Lightness can change faster than colour across a light pool
        d = np.sqrt((diff[..., 0] * 0.5) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)
        accept = ring & (d < step_tol)
        if not accept.any():
            break
        m[accept] = 1.0
    return m > 0


def fill_holes(mask: np.ndarray, blocked: np.ndarray, max_share: float = 0.02) -> np.ndarray:
    """Fill small gaps fully surrounded by wall, unless they are objects (frames, switches the model saw…)."""
    inv = (~mask).astype(np.uint8)
    n, ids, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > max_share * mask.size:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # touches the photo border: not enclosed
        hole = ids[y:y + bh, x:x + bw] == i
        if blocked[y:y + bh, x:x + bw][hole].mean() > 0.3:
            continue  # mostly an object: keep it unpainted
        out[y:y + bh, x:x + bw][hole] = True
    return out


def doubtful_gaps(mask: np.ndarray, lab: np.ndarray, edges: np.ndarray, max_share: float = 0.08,
                  min_smooth_border: float = 0.35, colour_gap: float = 10.0) -> np.ndarray:
    """
    Gaps in the wall the model got wrong. A real object (frame, curtain, lamp)
    has a visible outline wherever it meets the wall; a mislabelled blotch of
    wall blends into the wall around it with no outline, in the wall's colour.
    Gaps are cut along the photo's edges first, so a blotch touching the
    ceiling is judged on its own.
    """
    total = mask.size
    edge_zone = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    gaps = (~mask & ~edge_zone).astype(np.uint8)
    n, ids, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)
    out = np.zeros_like(mask)
    H, W = mask.shape
    k5 = np.ones((5, 5), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20 or area > max_share * total:
            continue
        if y <= 1:
            continue  # touches the top of the photo: usually a ceiling corner, not a blotch
        pad = 10
        y0, y1, x0, x1 = max(0, y - pad), min(H, y + h + pad), max(0, x - pad), min(W, x + w + pad)
        g = ids[y0:y1, x0:x1] == i
        m = mask[y0:y1, x0:x1]
        ez = edge_zone[y0:y1, x0:x1]
        ring = cv2.dilate(g.astype(np.uint8), k5).astype(bool) & ~g
        smooth = (ring & ~ez & m).sum() / max(int(ring.sum()), 1)   # share of its border that blends into wall
        if smooth < min_smooth_border:
            continue                       # outlined, or bordered by other things: a real object
        surround = (ring & (m | ez)).sum()        # wall, or a real edge (e.g. the ceiling line)
        if surround < 0.7 * ring.sum():
            continue                       # not enclosed by wall and edges
        inside = np.median(lab[y0:y1, x0:x1][g], axis=0)
        far = cv2.dilate(g.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & \
            ~cv2.dilate(g.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        around_px = lab[y0:y1, x0:x1][far & m]   # wall a few pixels away, not the blurry border
        if len(around_px) < 10:
            continue
        around = np.median(around_px, axis=0)
        gap = float(np.hypot(inside[1] - around[1], inside[2] - around[2]) + 0.25 * abs(inside[0] - around[0]))
        if gap < colour_gap:
            out[y0:y1, x0:x1] |= g
    # The edge pixels between two doubtful bits, or a doubtful bit and the wall, go with them
    if out.any():
        near = cv2.dilate(out.astype(np.uint8), k5).astype(bool)
        out |= near & edge_zone & ~mask & cv2.dilate((mask | out).astype(np.uint8), k5).astype(bool) & \
            ~cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return out


def snap_to_edges(mask: np.ndarray, image_rgb: np.ndarray, never: np.ndarray, band: int = 5) -> np.ndarray:
    """
    The model's border is only accurate to a few pixels. Re-decide every pixel
    within `band` of the border by flooding from the sure inside and the sure
    outside at once (watershed): they meet on the strongest edge in the photo.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    m8 = mask.astype(np.uint8)
    inside = cv2.erode(m8, k).astype(bool)
    # Thin wall strips narrower than the band keep a thin core so they aren't lost
    lost = mask & ~cv2.dilate(inside.astype(np.uint8), k).astype(bool)
    inside |= cv2.erode(lost.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    outside = cv2.erode((~mask).astype(np.uint8), k).astype(bool) | never
    markers = np.zeros(mask.shape, np.int32)
    markers[outside] = 1
    markers[inside & ~never] = 2
    smooth = cv2.bilateralFilter(image_rgb, 7, 30, 7)
    cv2.watershed(cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR), markers)
    out = markers == 2
    ridge = markers == -1
    out[ridge] = mask[ridge]
    return out


def tidy(mask: np.ndarray, min_share: float = 0.002) -> np.ndarray:
    """Remove specks and close hairline cracks."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, ids, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_share * mask.size
    return keep[ids]


def feather(mask: np.ndarray, image_rgb: np.ndarray, radius: int = 2) -> np.ndarray:
    """Soft 1–2 px edge that follows the photo (no halo, no jaggies)."""
    from src.colour_studio.segmentation import guided_filter

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    soft = guided_filter(gray, mask.astype(np.float32), radius=radius, eps=2e-3)
    soft = np.clip(soft, 0.0, 1.0)
    # Never let the feather eat into the wall or spread far outside it
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    outer = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    soft[inner] = 1.0
    soft[~outer] = 0.0
    return soft.astype(np.float32)


def refine_wall(image_rgb: np.ndarray, prob: np.ndarray, label_map: Optional[np.ndarray] = None,
                label_conf: Optional[np.ndarray] = None, id2label: Optional[Dict[int, str]] = None,
                return_parts: bool = False):
    """
    prob: wall probability (0..1) at the photo's working resolution.
    Returns a clean bool wall mask at the same resolution
    (with return_parts: (mask, edges, objects) for the per-wall steps).
    """
    lab = lab_image(image_rgb)
    edges = strong_edges(image_rgb)
    raw_objects = object_mask(label_map, label_conf, id2label or {})
    if raw_objects.shape != prob.shape:
        raw_objects = np.zeros_like(prob, dtype=bool)

    k3 = np.ones((3, 3), np.uint8)
    mask = (prob > 0.5) & ~cv2.dilate(raw_objects.astype(np.uint8), k3).astype(bool)
    mask = remove_leaks(mask, prob, lab, edges, raw_objects)   # first: so leaks can't vouch for blotches
    doubtful = doubtful_gaps(mask, lab, edges)          # blotches with no outline: really wall
    objects = cv2.dilate((raw_objects & ~doubtful).astype(np.uint8), k3).astype(bool)
    mask = (mask | doubtful) & ~objects
    mask = snap_to_edges(mask, image_rgb, never=objects & (prob < 0.5), band=8)
    # Growth: never across edges or into objects; keep away from things the
    # model is very sure are not wall.
    blocked = objects | cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    reach = int(0.15 * np.hypot(*prob.shape))
    mask = grow_smooth(mask, lab, blocked, max_steps=reach, min_prob=(prob > 0.02) | doubtful)
    mask = tidy(fill_holes(mask, objects))
    if return_parts:
        return mask, edges, outlined_only(objects, edges)
    return mask


def _fit_line(xs: np.ndarray, ys: np.ndarray, tol: float = 2.5, iters: int = 200, seed: int = 0):
    """Robust straight-line fit y = a*x + b (RANSAC, then least squares on the inliers)."""
    if len(xs) < 2:
        return None
    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(iters):
        i, j = rng.choice(len(xs), 2, replace=False)
        if xs[i] == xs[j]:
            continue
        a = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - a * xs[i]
        inl = np.abs(ys - (a * xs + b)) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (a, b), inl
    if best is None:
        return None
    a, b = np.polyfit(xs[best_in], ys[best_in], 1)
    return float(a), float(b), best_in


def outlined_only(objects: np.ndarray, edges: np.ndarray, min_support: float = 0.3) -> np.ndarray:
    """Keep only objects that have a visible outline in the photo (real ones)."""
    n, ids, stats, _ = cv2.connectedComponentsWithStats(objects.astype(np.uint8), connectivity=8)
    if n <= 1:
        return objects
    edge_zone = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    border = objects & ~cv2.erode(objects.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hits = np.bincount(ids[border & edge_zone], minlength=n)
    tot = np.maximum(np.bincount(ids[border], minlength=n), 1)
    keep = hits / tot >= min_support
    keep[0] = False
    return keep[ids]


def straighten_top(region: np.ndarray, edges: np.ndarray, blocked: np.ndarray, others: np.ndarray,
                   max_depth: float = 0.25) -> np.ndarray:
    """
    A wall meets the ceiling along a straight line. Where that line is visible
    (an edge right above the wall), fit it; where light washes it out, the wall
    stops short in a ragged bite — paint up to the fitted line there.
    """
    h, w = region.shape
    cols = np.nonzero(region.any(axis=0))[0]
    if len(cols) < 30:
        return region
    tops = region[:, cols].argmax(axis=0)
    usable = tops > 2                          # columns where the wall runs off the photo don't count
    if usable.sum() < 30:
        return region
    # The straight part of the outline is the ceiling line (seen as an edge, or
    # simply where the model's outline runs dead straight); bites bend away below it.
    fit = _fit_line(cols[usable].astype(np.float32), tops[usable].astype(np.float32))
    if fit is None:
        return region
    a, b, inl = fit
    span = np.ptp(cols[usable][inl]) if inl.sum() else 0
    if inl.sum() < max(25, 0.4 * usable.sum()) or span < 0.25 * np.ptp(cols):
        return region                          # no clear straight ceiling line
    out = region.copy()
    max_px = int(max_depth * h)
    for x, t in zip(cols, tops):
        y_line = int(round(a * x + b))
        y0 = max(0, y_line + 1)
        if t - y0 < 3 or t - y0 > max_px:
            continue
        seg = slice(y0, t)
        if blocked[seg, x].any() or others[seg, x].any():
            continue                           # an object or another wall is there
        if edges[y0 + 2:max(y0 + 2, t - 2), x].any():
            continue                           # a real edge inside (a beam, a dropped ceiling): leave it
        out[seg, x] = True
    return out
