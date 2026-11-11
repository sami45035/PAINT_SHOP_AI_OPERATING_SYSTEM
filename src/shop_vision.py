"""
Shop vision (Layer 5): what the camera learns about the shop, beyond boxes.

1. Footfall     — people crossing the door line (in / out), see camera_stream.
2. Buyers       — each visitor's journey: which shelves they stood at, whether
                  they reached the billing counter. Bills come from the sales
                  table, so "visitors who bought" = bills / visitors.
3. Theft        — a can leaves a shelf, the person who took it walks out
                  without going to the counter -> UNPAID_EXIT alert with photo.
                  Shelf picks are also reconciled with the day's bills.
4. Fast movers  — every can taken off a monitored shelf is a "pick"; picks,
                  put-backs and sales together show what moves and what people
                  pick up but don't buy.
5. Shelf stock  — cans are counted on each shelf slot from the picture, so the
                  app knows when a shelf needs refilling from the store room and
                  when the shop needs to pre-order.

Everything here is plain OpenCV on a CPU: no extra model to download.

Layout (all coordinates are fractions of the picture, so any camera size works):
    door:       {"y", "x1", "x2"}  door line — a person's feet crossing it
                downward = coming in, upward = leaving
    counter:    [x1, y1, x2, y2]   where customers stand to pay (feet inside)
    staff_zone: [x1, y1, x2, y2]   staff-only area (feet inside)
    shelves:    [{"id", "name", "box": [x1, y1, x2, y2], "product_id", "capacity"}]
                one slot = one product, cans standing side by side
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]          # x, y, w, h in pixels

# ----------------------------------------------------------------------------- layout

DEFAULT_LAYOUT: Dict[str, Any] = {
    "door": {"y": 0.68, "x1": 0.0, "x2": 1.0},
    "counter": [0.04, 0.55, 0.40, 0.95],
    "staff_zone": [0.68, 0.10, 0.96, 0.58],
    "shelves": [],
}

# The demo shop: what the drawn scene looks like (see DemoShop)
DEMO_SLOT_PRODUCTS = [  # slot id, shade, pack (litres)
    ("A1", "Sage Whisper", 1.0), ("A2", "Teal Serenity", 1.0), ("A3", "Crisp Linen", 1.0),
    ("B1", "Warm Beige", 4.0), ("B2", "Royale Blue", 4.0), ("B3", "Universal White Primer", 4.0),
]
_COLS = [(0.035, 0.125), (0.135, 0.225), (0.235, 0.325)]
_ROWS = {"A": (0.30, 0.47), "B": (0.53, 0.70)}


def demo_layout(products: Optional[Dict[Tuple[str, float], int]] = None) -> Dict[str, Any]:
    shelves = []
    for sid, shade, size in DEMO_SLOT_PRODUCTS:
        (x1, x2), (y1, y2) = _COLS[int(sid[1]) - 1], _ROWS[sid[0]]
        shelves.append({"id": sid, "name": f"Shelf {sid}", "box": [x1, y1, x2, y2],
                        "product_id": (products or {}).get((shade, size)), "capacity": 4,
                        "label": f"{shade} {size:g} L"})
    return {
        "door": {"y": 0.50, "x1": 0.38, "x2": 0.64},
        "counter": [0.60, 0.62, 0.97, 0.80],
        "staff_zone": [0.80, 0.42, 0.995, 0.60],
        "shelves": shelves,
    }


def demo_products_from_db() -> Dict[Tuple[str, float], int]:
    """(shade, pack) -> product id for the demo shelves."""
    import sqlite3

    from src.config import DB_PATH

    out: Dict[Tuple[str, float], int] = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for _, shade, size in DEMO_SLOT_PRODUCTS:
                row = conn.execute("SELECT id FROM products WHERE shade_name = ? AND pack_size_litres = ? LIMIT 1;",
                                   (shade, size)).fetchone()
                if row:
                    out[(shade, size)] = int(row[0])
    except sqlite3.Error:
        pass
    return out


def load_layout() -> Dict[str, Any]:
    """The real camera's layout (saved from the Set up tab)."""
    from src.database import get_setting

    saved = get_setting("camera_layout")
    layout = {**DEFAULT_LAYOUT, **(saved or {})}
    layout["door"] = {**DEFAULT_LAYOUT["door"], **(layout.get("door") or {})}
    return layout


def save_layout(layout: Dict[str, Any]) -> None:
    from src.database import set_setting

    set_setting("camera_layout", layout)


def to_px(box: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """[x1, y1, x2, y2] fractions -> pixel corners."""
    return int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)


def feet(b: Box) -> Tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h * 0.97


def inside(pt: Tuple[float, float], corners: Tuple[int, int, int, int]) -> bool:
    return corners[0] <= pt[0] <= corners[2] and corners[1] <= pt[1] <= corners[3]


def overlap_frac(b: Box, corners: Tuple[int, int, int, int]) -> float:
    """How much of the corners-box is covered horizontally by b (0..1)."""
    x, y, w, h = b
    x1, y1, x2, y2 = corners
    if y > y2 or y + h < y1:
        return 0.0
    inter = max(0, min(x + w, x2) - max(x, x1))
    return inter / max(1, x2 - x1)


def boxes_touch(b: Box, corners: Tuple[int, int, int, int], pad: int = 0) -> bool:
    x, y, w, h = b
    x1, y1, x2, y2 = corners
    return not (x + w < x1 - pad or x > x2 + pad or y + h < y1 - pad or y > y2 + pad)


# ----------------------------------------------------------------------------- shelves

BINS = 32


def column_profile(frame: np.ndarray, corners: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    How 'busy' each vertical strip of a shelf slot is. A can (outline, lid,
    label print) makes strong edges; an empty bit of shelf is flat. Returns
    BINS values (edge strength), or None if the slot is off the picture.
    """
    x1, y1, x2, y2 = corners
    h, w = frame.shape[:2]
    x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    rw = 128
    rh = max(24, int(round(gray.shape[0] * rw / gray.shape[1])))
    gray = cv2.GaussianBlur(cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA), (3, 3), 0).astype(np.float32)
    mag = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    # Ignore the top and bottom edge of the slot (shelf boards)
    band = mag[int(rh * 0.08): int(rh * 0.92)]
    cols = band.mean(axis=0)
    return cols.reshape(BINS, -1).mean(axis=1)


@dataclass
class SlotState:
    cans: Optional[int] = None           # committed count
    readings: List[int] = field(default_factory=list)
    threshold: float = 8.0               # edge strength that means "something here"
    occupied_at_cal: Optional[int] = None
    cans_at_cal: Optional[int] = None
    estimated: bool = True               # True until calibrated


class ShelfMonitor:
    """Counts cans per shelf slot and turns changes into pick / restock events."""

    STABLE = 3                            # readings in a row before a change counts

    def __init__(self, slots: List[Dict[str, Any]]):
        self.slots: List[Dict[str, Any]] = []
        self.state: Dict[str, SlotState] = {}
        self.set_slots(slots)

    def set_slots(self, slots: List[Dict[str, Any]]) -> None:
        old = self.state
        self.slots = list(slots or [])
        self.state = {s["id"]: old.get(s["id"], SlotState()) for s in self.slots}

    def calibrate(self, frame: np.ndarray, counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        """
        'The shelves look like this now': remember each slot's look with its
        known number of cans (default: full).
        """
        h, w = frame.shape[:2]
        done = {}
        for s in self.slots:
            prof = column_profile(frame, to_px(s["box"], w, h))
            if prof is None:
                continue
            st = self.state[s["id"]]
            n = int((counts or {}).get(s["id"], s.get("capacity", 4)))
            st.threshold = max(6.0, 0.35 * float(np.percentile(prof, 90)))
            st.occupied_at_cal = int((prof > st.threshold).sum())
            st.cans_at_cal = n
            st.cans, st.readings, st.estimated = n, [], False
            done[s["id"]] = n
        return done

    def read(self, frame: np.ndarray, slot: Dict[str, Any]) -> Optional[int]:
        h, w = frame.shape[:2]
        prof = column_profile(frame, to_px(slot["box"], w, h))
        if prof is None:
            return None
        st = self.state[slot["id"]]
        cap = int(slot.get("capacity", 4))
        occ = int((prof > st.threshold).sum())
        if st.occupied_at_cal and st.cans_at_cal:
            n = st.cans_at_cal * occ / st.occupied_at_cal
        else:                               # not calibrated: cans fill ~85% of a full slot's width
            n = cap * occ / (0.85 * BINS)
        return int(max(0, min(cap, round(n))))

    def update(self, frame: np.ndarray, people: List[Box]) -> List[Tuple[Dict[str, Any], int, int]]:
        """
        One frame. Slots hidden behind a person are skipped (a hand in front of
        the shelf isn't a missing can). Returns [(slot, change, cans_now)] for
        counts that changed and held steady for STABLE readings.
        """
        h, w = frame.shape[:2]
        pad = int(0.01 * w)
        changes = []
        for s in self.slots:
            st = self.state[s["id"]]
            corners = to_px(s["box"], w, h)
            if any(boxes_touch(b, corners, pad) for b in people):
                st.readings = []
                continue
            n = self.read(frame, s)
            if n is None:
                continue
            st.readings = (st.readings + [n])[-self.STABLE:]
            if len(st.readings) < self.STABLE or len(set(st.readings)) != 1:
                continue
            if st.cans is None:
                st.cans = n                 # first steady look: the starting point, not a change
                continue
            if n != st.cans:
                changes.append((s, n - st.cans, n))
                st.cans = n
        return changes

    def counts(self) -> Dict[str, Optional[int]]:
        return {sid: st.cans for sid, st in self.state.items()}


# ----------------------------------------------------------------------------- journeys

@dataclass
class Journey:
    """One customer's visit, from the door line in to the door line out."""
    entered_at: Any = None               # datetime
    entered_t: float = 0.0
    counter_s: float = 0.0
    reached_counter: bool = False
    near_since: Dict[str, float] = field(default_factory=dict)   # slot id -> last time seen at it
    near_s: Dict[str, float] = field(default_factory=dict)       # slot id -> seconds at it
    picks: Dict[str, int] = field(default_factory=dict)          # slot id -> cans taken (net)
    put_back: int = 0
    in_staff_zone: bool = False
    counter_from: Any = None             # first / last moment at the billing counter (datetime) — bills are matched to this
    counter_to: Any = None

    @property
    def shelves(self) -> List[str]:
        return sorted(sid for sid, s in self.near_s.items() if s >= 0.8 or self.picks.get(sid))

    @property
    def net_picked(self) -> int:
        return max(0, sum(self.picks.values()))

    def outcome(self) -> str:
        if self.reached_counter:
            return "reached_counter"
        if self.net_picked > 0:
            return "unpaid_exit"
        return "browsed"


COUNTER_SECONDS = 1.5      # standing at the counter this long = went to pay / ask
PICK_WINDOW_S = 20.0       # a can that disappears is credited to whoever stood there in the last 20 s
PUT_BACK_WINDOW_S = 180.0  # a can coming back within 3 min from the same person = put back


def credit_pick(journeys: Dict[int, Journey], slot_id: str, now_t: float) -> Optional[int]:
    """Which person most recently stood at this slot (within PICK_WINDOW_S)?"""
    best, best_t = None, -1.0
    for tid, j in journeys.items():
        t = j.near_since.get(slot_id)
        if t is not None and now_t - t <= PICK_WINDOW_S and t > best_t:
            best, best_t = tid, t
    return best


def credit_put_back(journeys: Dict[int, Journey], slot_id: str, now_t: float) -> Optional[int]:
    """A can returned to a slot by someone still in the shop who took one from it recently."""
    best, best_t = None, -1.0
    for tid, j in journeys.items():
        t = j.near_since.get(slot_id)
        if j.picks.get(slot_id, 0) > 0 and t is not None and now_t - t <= PUT_BACK_WINDOW_S and t > best_t:
            best, best_t = tid, t
    return best


# ----------------------------------------------------------------------------- demo shop

def _lerp(path: List[Tuple[float, float, float]], t: float) -> Optional[Tuple[float, float]]:
    if t < path[0][0] or t > path[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
    return path[-1][1], path[-1][2]


class DemoShop:
    """
    A drawn paint shop on a 64-second loop, analysed by the same code as a real
    camera. What happens (and what the app should conclude):
      0-24 s  Buyer: takes a can of Royale Blue 4 L (B2), pays at the counter, leaves
      12-36 s Browser: picks Teal Serenity 1 L (A2), reads it, puts it back, leaves
      37-49 s Walk-out: takes Warm Beige 4 L (B1), leaves WITHOUT going to the counter
      45-57 s Enquiry: goes straight to the counter, asks, leaves
      56-64 s Staff: comes out of the store room and refills the shelves
    """

    W, H, LOOP = 960, 540, 64.0
    START = {"A1": 1, "A2": 3, "A3": 4, "B1": 4, "B2": 4, "B3": 2}
    SHELF_EVENTS = [(9.0, "B2", -1), (19.0, "A2", -1), (28.0, "A2", +1), (39.5, "B1", -1),
                    (60.0, "B1", +1), (60.0, "B2", +1)]
    ACTORS = [
        ("buyer", (52, 101, 191), [(0, .51, .42), (4, .50, .62), (7, .18, .86), (11, .18, .86), (14, .72, .72),
                                   (19, .72, .72), (22, .51, .56), (24, .51, .42)]),
        ("browser", (60, 150, 90), [(12, .52, .42), (16, .30, .70), (18, .18, .86), (20, .18, .86), (21, .42, .80),
                                    (25, .42, .80), (27, .18, .86), (29, .18, .86), (31, .42, .75), (34, .50, .55),
                                    (36, .50, .42)]),
        ("walkout", (40, 40, 160), [(37, .49, .42), (40, .08, .86), (42, .08, .86), (44, .30, .70), (47, .47, .55),
                                    (49, .47, .42)]),
        ("enquiry", (150, 90, 160), [(45, .54, .42), (49, .74, .72), (52, .74, .72), (55, .54, .55), (57, .54, .42)]),
        ("staff", (30, 120, 200), [(56, .92, .50), (59, .13, .86), (61, .13, .86), (63.9, .92, .50)]),
    ]

    def __init__(self, layout: Dict[str, Any]):
        self.layout = layout
        self.labels = {s["id"]: s.get("label", s["name"]) for s in layout["shelves"]}
        self.colours = {}
        try:
            import sqlite3

            from src.config import DB_PATH

            with sqlite3.connect(DB_PATH) as conn:
                for s in layout["shelves"]:
                    if s.get("product_id"):
                        row = conn.execute("SELECT hex_code FROM products WHERE id = ?;", (s["product_id"],)).fetchone()
                        if row:
                            hx = row[0].lstrip("#")
                            self.colours[s["id"]] = tuple(int(hx[i:i + 2], 16) for i in (4, 2, 0))  # BGR
        except Exception:  # noqa: BLE001
            pass
        self._background = self._draw_background()

    # ---- the script
    def cans_at(self, t: float) -> Dict[str, int]:
        cans = dict(self.START)
        for et, sid, d in self.SHELF_EVENTS:
            if t >= et:
                cans[sid] += d
        return cans

    def people_at(self, t: float) -> List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int]]]:
        out = []
        for name, colour, path in self.ACTORS:
            p = _lerp(path, t)
            if p is None:
                continue
            fx, fy = p[0] * self.W, p[1] * self.H
            bh = self.H * (0.20 + 0.45 * max(0.0, fy / self.H - 0.40) / 0.6)
            bw = 0.36 * bh
            out.append((name, (int(fx - bw / 2), int(fy - bh), int(bw), int(bh)), colour))
        return out

    # ---- drawing
    def _draw_background(self) -> np.ndarray:
        W, H = self.W, self.H
        f = np.zeros((H, W, 3), np.uint8)
        wall_y = int(0.40 * H)
        f[:wall_y] = (178, 196, 210)                                       # back wall
        for y in range(wall_y, H):                                         # floor with depth shading
            k = (y - wall_y) / (H - wall_y)
            f[y] = (int(120 + 40 * k), int(140 + 40 * k), int(150 + 40 * k))
        for i in range(-8, 18):                                            # floor tiles
            x0 = int(W / 2 + i * 0.09 * W * 0.35)
            x1 = int(W / 2 + i * 0.09 * W * 1.6)
            cv2.line(f, (x0, wall_y), (x1, H), (150, 165, 175), 1, cv2.LINE_AA)
        # entrance
        cv2.rectangle(f, (int(.44 * W), int(.08 * H)), (int(.58 * W), wall_y), (95, 120, 140), -1)
        cv2.rectangle(f, (int(.44 * W), int(.08 * H)), (int(.58 * W), wall_y), (60, 70, 80), 3)
        cv2.putText(f, "ENTRANCE", (int(.455 * W), int(.07 * H)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 70, 80), 1, cv2.LINE_AA)
        # store room (staff only)
        cv2.rectangle(f, (int(.86 * W), int(.12 * H)), (int(.97 * W), int(.44 * H)), (70, 95, 125), -1)
        cv2.putText(f, "STORE", (int(.875 * W), int(.10 * H)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 70, 80), 1, cv2.LINE_AA)
        # shelf unit
        cv2.rectangle(f, (int(.025 * W), int(.27 * H)), (int(.335 * W), int(.74 * H)), (70, 90, 115), -1)
        for _, (y1, y2) in _ROWS.items():
            cv2.rectangle(f, (int(.03 * W), int(y1 * H)), (int(.33 * W), int(y2 * H)), (206, 214, 220), -1)
            cv2.rectangle(f, (int(.025 * W), int(y2 * H)), (int(.335 * W), int(y2 * H) + 7), (60, 78, 100), -1)
        # counter
        cv2.rectangle(f, (int(.62 * W), int(.80 * H)), (int(.97 * W), H), (70, 100, 140), -1)
        cv2.rectangle(f, (int(.62 * W), int(.78 * H)), (int(.97 * W), int(.82 * H)), (55, 75, 105), -1)
        cv2.rectangle(f, (int(.86 * W), int(.70 * H)), (int(.93 * W), int(.78 * H)), (40, 40, 40), -1)   # till
        cv2.putText(f, "BILLING", (int(.70 * W), int(.93 * H)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 235, 240), 2, cv2.LINE_AA)
        return f

    def _draw_can(self, f: np.ndarray, x: int, y_bottom: int, cw: int, ch: int, colour, label: str) -> None:
        x2, top = x + cw, y_bottom - ch
        cv2.rectangle(f, (x, top), (x2, y_bottom), (200, 205, 210), -1)                 # tin
        cv2.rectangle(f, (x, top + int(ch * .22)), (x2, y_bottom - int(ch * .12)), colour, -1)  # colour band
        cv2.rectangle(f, (x + 3, top + int(ch * .40)), (x2 - 3, top + int(ch * .62)), (250, 250, 250), -1)  # label
        cv2.putText(f, label[:2].upper(), (x + 4, top + int(ch * .58)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.rectangle(f, (x, top), (x2, y_bottom), (70, 70, 70), 1)
        cv2.line(f, (x, top + 3), (x2, top + 3), (120, 120, 120), 1)                   # lid rim

    def _draw_person(self, f: np.ndarray, box, colour, staff: bool = False) -> None:
        x, y, w, h = box
        cx = x + w // 2
        r = max(6, int(h * 0.075))
        cv2.circle(f, (cx, y + r + 2), r, (150, 175, 200), -1, cv2.LINE_AA)
        body_top, body_bot = y + 2 * r + 4, y + int(h * 0.60)
        cv2.rectangle(f, (cx - w // 2 + 4, body_top), (cx + w // 2 - 4, body_bot), colour, -1)
        if staff:
            cv2.putText(f, "STAFF", (cx - w // 2 + 4, body_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        lw = max(3, w // 6)
        cv2.line(f, (cx - w // 6, body_bot), (cx - w // 5, y + h), (50, 50, 60), lw)
        cv2.line(f, (cx + w // 6, body_bot), (cx + w // 5, y + h), (50, 50, 60), lw)

    def frame(self, t: float) -> Tuple[np.ndarray, List[Box]]:
        """(picture, people boxes) at loop time t."""
        t = t % self.LOOP
        f = self._background.copy()
        W, H = self.W, self.H
        cans = self.cans_at(t)
        for s in self.layout["shelves"]:
            x1, y1, x2, y2 = to_px(s["box"], W, H)
            cap = int(s.get("capacity", 4))
            cw = int((x2 - x1 - 8) / cap) - 3
            ch = int((y2 - y1) * 0.78)
            for i in range(cans.get(s["id"], 0)):
                self._draw_can(f, x1 + 4 + i * (cw + 3), y2 - 2, cw, ch, self.colours.get(s["id"], (90, 120, 200)),
                               self.labels.get(s["id"], s["id"]))
            cv2.putText(f, s["id"], (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
        people = sorted(self.people_at(t), key=lambda p: p[1][1] + p[1][3])   # far people first
        for name, box, colour in people:
            self._draw_person(f, box, colour, staff=(name == "staff"))
        noise = np.random.default_rng(int(t * 10)).normal(0, 2.0, f.shape)   # a little sensor noise
        f = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return f, [b for _, b, _ in people]


def draw_layout(frame: np.ndarray, layout: Dict[str, Any], pending: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """Zones drawn on a still, for the Set up tab (BGR in, BGR out)."""
    out = frame.copy()
    h, w = out.shape[:2]

    def tag(text, x, y, colour):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 8, y + 2), (32, 26, 21), -1)
        cv2.putText(out, text, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    d = layout["door"]
    y = int(d["y"] * h)
    cv2.line(out, (int(d.get("x1", 0) * w), y), (int(d.get("x2", 1) * w), y), (74, 192, 245), 3)
    tag("Door line (coming in = crossing downward)", int(d.get("x1", 0) * w) + 4, y - 6, (74, 192, 245))
    for key, name, col in (("counter", "Billing counter", (235, 140, 60)), ("staff_zone", "Staff only", (102, 112, 249))):
        x1, y1, x2, y2 = to_px(layout[key], w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        tag(name, x1 + 2, y1 + 18, col)
    for s in layout.get("shelves", []):
        x1, y1, x2, y2 = to_px(s["box"], w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (128, 222, 74), 2)
        tag(f"{s['id']} · {s.get('label', '')}", x1 + 2, y2 - 2, (128, 222, 74))
    for px, py in pending or []:
        cv2.circle(out, (int(px * w), int(py * h)), 7, (255, 255, 255), -1)
        cv2.circle(out, (int(px * w), int(py * h)), 7, (20, 20, 20), 2)
    return out


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
