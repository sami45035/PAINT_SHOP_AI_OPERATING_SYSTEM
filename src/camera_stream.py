"""
Store Camera (Layer 5): the camera as a data source for the shop.

Sources
- "webcam": laptop / USB camera (the app reads it when running on that laptop;
            on Streamlit Cloud the browser sends frames instead)
- "esp32":  ESP32-CAM MJPEG stream on the shop Wi-Fi
- "demo":   a drawn paint shop (src.shop_vision.DemoShop). Its data is saved
            with source='demo' so it never mixes with real numbers.

Per frame
- Finds people (OpenCV HOG detector + non-maximum suppression; faces for people
  close to the camera) and follows each one between frames.
- Footfall: feet crossing the door line, downward = in, upward = out.
- Journeys: for each customer who came in, which shelves they stood at, the
  cans they took or put back, and whether they reached the billing counter.
  Written to `visits` when they leave.
- Shelves: cans counted on each shelf slot (shop_vision.ShelfMonitor); a drop
  is a pick credited to the person who stood there, a rise is a put-back or a
  refill. Written to `shelf_events` / `shelf_status`.
- Theft: someone leaving with a picked can who never reached the counter
  (UNPAID_EXIT), movement after closing (AFTER_HOURS_MOTION), a customer in the
  staff-only area (RESTRICTED_ZONE_ENTRY). Saved with a photo, at most one per
  type per SECURITY_ALERT_COOLDOWN_SECONDS (each walk-out is its own alert).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src import clock
from src.config import (
    DATA_DIR,
    DEFAULT_CAMERA_INDEX,
    DOOR_LINE_Y,
    ESP32_DEFAULT_STREAM_URL,
    RESTRICTED_ZONE_BOX,
    SECURITY_ALERT_COOLDOWN_SECONDS,
)
from src.database import log_footfall, log_security_event, log_shelf_event, log_visit, set_shelf_status
from src.shop_vision import (
    COUNTER_SECONDS,
    DemoShop,
    Journey,
    ShelfMonitor,
    credit_pick,
    credit_put_back,
    demo_layout,
    demo_products_from_db,
    feet,
    inside,
    load_layout,
    overlap_frac,
    to_px,
)

SNAPSHOT_DIR = DATA_DIR / "security_snapshots"
SOURCE_LABELS = {"webcam": "Webcam", "esp32": "ESP32-CAM", "demo": "Demo shop"}
BLUE = (235, 140, 60)
LOST_JOURNEY_S = 600.0     # a customer the camera lost and never saw leaving is closed after 10 minutes
PRESENCE_CONFIRM_S = 1.5   # "people in view" mode: in view this long = a visitor (not someone walking past)
PRESENCE_GONE_S = 4.0      # ... and out of view this long = left
COUNT_MODES = {
    "presence": "People in view — camera at the counter or desk (counts each person it sees)",
    "door": "Door line — camera sees the whole doorway (counts feet crossing the line)",
}


def _tag(frame: np.ndarray, text: str, x: int, y: int, colour) -> None:
    """Readable label: coloured text on a dark tag."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = max(0, min(frame.shape[1] - tw - 8, x))
    y = max(th + 6, y)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + 3), DARK, -1)
    cv2.putText(frame, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)     # a customer the camera lost and never saw leaving is closed after 10 minutes

# Colours (BGR) for overlays
YELLOW, RED, GREEN, WHITE, DARK = (74, 192, 245), (102, 112, 249), (128, 222, 74), (255, 255, 255), (43, 32, 21)


def _reachable(url: str, timeout: float = 2.0) -> bool:
    """Quick TCP check so an offline ESP32 doesn't freeze the page for 30 s inside OpenCV."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(url)
    if not u.hostname:
        return False
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class Track:
    id: int
    centroid: Tuple[float, float]
    box: Tuple[int, int, int, int]
    last_seen: float
    frames: int = 1
    side: Optional[str] = None        # "out" (beyond the door line) or "in" (inside the shop)
    counted_in: bool = False
    counted_out: bool = False
    history: List[float] = field(default_factory=list)
    out_x: Optional[float] = None     # where the feet were when last seen beyond the door line
    first_seen: float = 0.0


class StoreVisionEngine:
    def __init__(self, source_type: str = "webcam", esp32_url: str = ESP32_DEFAULT_STREAM_URL,
                 log_to_db: Optional[bool] = None):
        if source_type == "synthetic":  # old name
            source_type = "demo"
        self.source_type = source_type
        self.esp32_url = esp32_url or ESP32_DEFAULT_STREAM_URL
        self.data_source = "demo" if source_type == "demo" else "camera"   # how rows are tagged in the database
        self.log_to_db = True if log_to_db is None else bool(log_to_db)
        self.cap: Optional[cv2.VideoCapture] = None
        self.connected = False
        self.status_message = ""

        # Zones: door line, counter, staff-only area, shelf slots (fractions of the picture)
        if source_type == "demo":
            self.layout = demo_layout(demo_products_from_db())
        else:
            try:
                self.layout = load_layout()
            except Exception:  # noqa: BLE001 - no database yet
                self.layout = {"door": {"y": DOOR_LINE_Y, "x1": 0.0, "x2": 1.0}, "counter": [0.04, 0.55, 0.40, 0.95],
                               "staff_zone": list(RESTRICTED_ZONE_BOX), "shelves": []}
        self.shelves = ShelfMonitor(self.layout.get("shelves", []))
        self.journeys: Dict[int, Journey] = {}           # track id -> customer inside the shop
        self.lost: List[Tuple[Journey, float]] = []       # customers the tracker lost, waiting to be seen leaving
        self.events: List[Dict[str, Any]] = []            # newest first, for the Live screen
        self._saved_shelves: Dict[str, Optional[int]] = {}
        self._crossed: List[Tuple[Track, str]] = []
        self._det_tracks: List[int] = []
        self._frame_w: Optional[int] = None
        self._demo_boxes: List[Tuple[int, int, int, int]] = []

        # Session counters (the database keeps the day's totals for real sources)
        self.total_entered = 0
        self.total_exited = 0
        self.current_occupancy = 0

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.last_alert_at: Dict[str, float] = {}
        self.last_alert: Optional[Dict[str, Any]] = None
        self.latest_raw: Optional[np.ndarray] = None      # last clean frame (for zone set-up and calibration)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
        self.frames_seen = 0

        # Faces: so someone close to the camera (at the counter) is still seen
        self.face = self.face_side = None
        try:
            base = Path(cv2.data.haarcascades)
            for attr, name in (("face", "haarcascade_frontalface_default.xml"), ("face_side", "haarcascade_profileface.xml")):
                if (base / name).exists():
                    c = cv2.CascadeClassifier(str(base / name))
                    setattr(self, attr, None if c.empty() else c)
        except Exception:
            self.face = self.face_side = None

        # Background reading: the camera is read continuously so people walking
        # through the door aren't missed between screen refreshes.
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest: Tuple[Optional[np.ndarray], Dict[str, Any]] = (None, {})

        self.hog = None
        if hasattr(cv2, "HOGDescriptor"):
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self.hog = None

        # Demo shop
        self._demo_t0 = time.time()
        self.demo_shop = DemoShop(self.layout) if source_type == "demo" else None

    # ---------------------------------------------------------------- zones
    @property
    def door_line_y(self) -> float:
        return float(self.layout["door"]["y"])

    @door_line_y.setter
    def door_line_y(self, v: float) -> None:
        self.layout["door"]["y"] = float(v)

    @property
    def zone_box(self) -> List[float]:
        return list(self.layout["staff_zone"])

    @zone_box.setter
    def zone_box(self, v) -> None:
        self.layout["staff_zone"] = [float(x) for x in v]

    @property
    def count_mode(self) -> str:
        return self.layout.get("count_mode") or ("presence" if self.source_type == "webcam" else "door")

    def set_layout(self, layout: Dict[str, Any]) -> None:
        self.layout = layout
        self.shelves.set_slots(layout.get("shelves", []))

    def _event(self, kind: str, text: str) -> None:
        self.events.insert(0, {"time": clock.fmt_time(clock.now()), "kind": kind, "text": text})
        del self.events[30:]

    def slot(self, slot_id: str) -> Dict[str, Any]:
        return next((s for s in self.layout.get("shelves", []) if s["id"] == slot_id), {"id": slot_id, "name": slot_id})

    # ---------------------------------------------------------------- stream
    def open_stream(self) -> bool:
        if self.source_type == "demo":
            self.connected = True
            self.status_message = "Demo feed — a drawn scene, not a real camera. Numbers are examples and nothing is saved."
            return True
        self.close_stream()
        if self.source_type == "esp32" and not _reachable(self.esp32_url):
            self.connected = False
            self.status_message = (f"Couldn't reach the ESP32-CAM at {self.esp32_url}. "
                                   "Check it is powered and on the same Wi-Fi as this computer.")
            return False
        target = self.esp32_url if self.source_type == "esp32" else DEFAULT_CAMERA_INDEX
        cap = cv2.VideoCapture(target)
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap = cap
                self.connected = True
                self.status_message = f"{SOURCE_LABELS[self.source_type]} connected — counting and guard are active."
                return True
            cap.release()
        self.connected = False
        if self.source_type == "esp32":
            self.status_message = (f"Couldn't reach the ESP32-CAM at {self.esp32_url}. "
                                   "Check it is powered and on the same Wi-Fi as this computer.")
        else:
            self.status_message = ("No webcam found. On Streamlit Cloud the server can't see your laptop's camera — "
                                   "run the app on the laptop, or use the browser camera below.")
        return False

    # ---------------------------------------------------------------- background
    def start(self, fps: float = 8.0) -> None:
        """Keep reading and analysing frames in the background until close_stream()."""
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.connected and not self.open_stream():
            return
        self._stop.clear()

        def loop():
            gap = 1.0 / fps
            while not self._stop.is_set():
                t0 = time.time()
                frame = self.read_frame()
                if frame is None:
                    if self.source_type != "demo":
                        break                       # camera gone: the screen shows the message
                    continue
                annotated, metrics = self.process_frame(frame)
                with self._lock:
                    self.latest = (annotated, metrics)
                self._stop.wait(max(0.0, gap - (time.time() - t0)))

        self._thread = threading.Thread(target=loop, name=f"camera-{self.source_type}", daemon=True)
        self._thread.start()

    def snapshot(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Latest analysed frame from the background reader."""
        with self._lock:
            return self.latest

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close_stream(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        if self.source_type != "demo":
            self.connected = False

    def read_frame(self) -> Optional[np.ndarray]:
        if self.source_type == "demo":
            return self.generate_synthetic_store_frame()
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.connected = False
            self.status_message = f"{SOURCE_LABELS[self.source_type]} stopped sending video."
            return None
        return frame

    # ---------------------------------------------------------------- demo
    def generate_synthetic_store_frame(self) -> np.ndarray:
        frame, self._demo_boxes = self.demo_shop.frame(time.time() - self._demo_t0)
        return frame

    # ---------------------------------------------------------------- detection
    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Whole people (standing/walking). Used for door counting and the staff-only zone."""
        if self.source_type == "demo":
            return list(self._demo_boxes)
        if self.hog is None:
            return []
        h, w = frame.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        try:
            boxes, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        except Exception:
            return []
        if len(boxes) == 0:
            return []
        weights = np.asarray(weights).ravel().tolist()
        keep = cv2.dnn.NMSBoxes([list(map(int, b)) for b in boxes], weights, 0.5, 0.45)
        keep = np.asarray(keep).ravel().tolist() if len(keep) else []
        return [tuple(int(v / scale) for v in boxes[i]) for i in keep]

    def detect_faces(self, frame: np.ndarray, bodies) -> List[Tuple[int, int, int, int]]:
        """Faces not already inside a body box: people close to the camera."""
        if self.face is None or self.source_type == "demo":
            return []
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        h, w = gray.shape
        min_side = max(40, w // 16)
        found = list(self.face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(min_side, min_side)))
        if self.face_side is not None:          # a head turned to the side (both directions)
            found += list(self.face_side.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(min_side, min_side)))
            flipped = self.face_side.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.15, minNeighbors=6,
                                                      minSize=(min_side, min_side))
            found += [(w - x - fw, y, fw, fh) for (x, y, fw, fh) in flipped]
        faces = []
        for f in sorted(found, key=lambda b: -b[2] * b[3]):      # keep one box per face
            fx, fy, fw_, fh_ = f
            if not any(abs((fx + fw_ / 2) - (gx + gw / 2)) < gw / 2 and abs((fy + fh_ / 2) - (gy + gh / 2)) < gh / 2
                       for gx, gy, gw, gh in faces):
                faces.append(tuple(int(v) for v in f))
        out = []
        for (x, y, fw, fh) in faces:
            cx, cy = x + fw / 2, y + fh / 2
            if any(bx <= cx <= bx + bw and by <= cy <= by + bh for bx, by, bw, bh in bodies):
                continue
            out.append((int(x), int(y), int(fw), int(fh)))
        return out

    def _update_tracks(self, detections, line_y: int, now_s: float, frame_h: int = 480) -> Tuple[int, int]:
        """
        Match detections to tracks and count door-line crossings. A person is
        counted only after being seen clearly on one side and then clearly on
        the other (a margin either side of the line), at most once each way —
        so someone standing on the line isn't counted over and over.
        """
        entered = exited = 0
        margin = 0.05 * frame_h
        self._crossed, self._det_tracks = [], []
        door = self.layout.get("door", {})
        x_lo = x_hi = None
        if self._frame_w:
            x_lo = (door.get("x1", 0.0) - 0.03) * self._frame_w
            x_hi = (door.get("x2", 1.0) + 0.03) * self._frame_w
        unmatched = set(self.tracks)
        for (x, y, bw, bh) in detections:
            feet_visible = y + bh < frame_h - 3            # a box cut off by the frame bottom can't tell
            c = (x + bw / 2.0, y + bh * 0.9)
            best, best_d = None, max(80.0, 0.6 * bh)
            for tid in unmatched:
                d = math.dist(c, self.tracks[tid].centroid)
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                tr = Track(self.next_track_id, c, (x, y, bw, bh), now_s, history=[c[1]], first_seen=now_s)
                self.tracks[tr.id] = tr
                self.next_track_id += 1
            else:
                unmatched.discard(best)
                tr = self.tracks[best]
                tr.centroid, tr.box, tr.last_seen = c, (x, y, bw, bh), now_s
                tr.frames += 1
                tr.history = (tr.history + [c[1]])[-20:]
            self._det_tracks.append(tr.id)
            if self.count_mode == "presence":
                # a person who stays in view a moment is a visitor; feet and door line don't matter
                if (not tr.counted_in or tr.counted_out) and tr.frames >= 2 and now_s - tr.first_seen >= PRESENCE_CONFIRM_S:
                    entered += 1
                    tr.counted_in, tr.counted_out = True, False
                    self._crossed.append((tr, "in"))
                continue
            if not feet_visible:
                continue
            side = "out" if c[1] < line_y - margin else ("in" if c[1] > line_y + margin else None)
            if side is None:
                continue
            # Only crossings through the doorway count (not staff walking along the back wall):
            # the street side of the crossing must be within the door's width.
            at_door = (lambda xx: xx is None or x_lo is None or x_lo <= xx <= x_hi)
            if side == "out":
                tr.out_x = c[0]
            if tr.side == "out" and side == "in" and tr.frames >= 2 and (not tr.counted_in or tr.counted_out) \
                    and at_door(tr.out_x):
                # a new entry — also when someone appears at the doorway right where a leaver vanished
                entered += 1
                tr.counted_in, tr.counted_out = True, False
                self._crossed.append((tr, "in"))
            elif tr.side == "in" and side == "out" and tr.frames >= 2 and not tr.counted_out and at_door(c[0]):
                exited += 1
                tr.counted_out = True
                self._crossed.append((tr, "out"))
            tr.side = side
        presence = self.count_mode == "presence"
        stale = PRESENCE_GONE_S if presence else (4.0 if self.source_type == "demo" else 1.5)
        for tid in [t for t, tr in self.tracks.items() if now_s - tr.last_seen > stale]:
            tr = self.tracks[tid]
            if presence:
                if tr.counted_in and not tr.counted_out:      # out of view long enough: they left
                    exited += 1
                    tr.counted_out = True
                    self._crossed.append((tr, "out"))
            else:
                j = self.journeys.pop(tid, None)
                if j is not None:
                    self.lost.append((j, now_s))           # may be the same person seen again at the door
            del self.tracks[tid]
        return entered, exited

    # ---------------------------------------------------------------- journeys
    def _finish_visit(self, j: Journey, now: datetime, frame: Optional[np.ndarray], seen_leaving: bool = True) -> None:
        outcome = j.outcome() if seen_leaving else ("reached_counter" if j.reached_counter else "browsed")
        picked = j.net_picked
        # how long they stayed, by the camera's own clock (the same clock that tracks them)
        stay = max(0.0, time.time() - j.entered_t) if j.entered_t else 0.0
        if self.count_mode == "presence" and seen_leaving:
            stay = max(0.0, stay - PRESENCE_GONE_S)    # they left when last seen, not when we gave up waiting
        entered = j.entered_at or now.replace(tzinfo=None)
        if self.log_to_db:
            log_visit(self.data_source, entered, entered + timedelta(seconds=stay),
                      j.reached_counter, j.shelves, picked, j.put_back, outcome)
        words = {"reached_counter": "went to the counter", "browsed": "looked around", "unpaid_exit": "left with a can"}
        stay_txt = f"{stay / 60:.0f} min" if stay >= 90 else f"{stay:.0f} s"
        self._event("visit", f"Customer left after {stay_txt} — {words.get(outcome, outcome)}"
                    + (f", took {picked} can{'s' if picked != 1 else ''}" if picked else "")
                    + (f", put back {j.put_back}" if j.put_back else ""))
        if outcome == "unpaid_exit" and frame is not None:
            items = ", ".join(f"{n} × {self.slot(sid).get('label') or self.slot(sid).get('name')} ({self.slot(sid).get('name')})"
                              for sid, n in j.picks.items() if n > 0)
            self._maybe_alert("UNPAID_EXIT", "Front door",
                              f"Someone took {items} and left at {clock.fmt_time(now)} without going to the counter.",
                              frame, now, key=f"UNPAID_EXIT:{id(j)}")

    def _adopt_lost(self, now_s: float) -> Optional[Journey]:
        """Someone leaving whom the tracker didn't see come in: most likely the customer it lost track of."""
        if not self.lost:
            return None
        j, _ = self.lost.pop()          # most recently lost
        return j

    def _update_journeys(self, frame: np.ndarray, people, now: datetime, now_s: float, after_hours: bool):
        h, w = frame.shape[:2]
        for tr, direction in self._crossed:
            if direction == "in":
                since = tr.first_seen if self.count_mode == "presence" else now_s   # presence: they arrived when first seen
                self.journeys[tr.id] = Journey(entered_at=now.replace(tzinfo=None) - timedelta(seconds=now_s - since),
                                               entered_t=since)
                self._event("door", "Customer came in")
            else:
                j = self.journeys.pop(tr.id, None) or self._adopt_lost(now_s)
                if j is not None:
                    self._finish_visit(j, now, frame)
                else:
                    self._event("door", "Someone left")
        # customers the tracker lost long ago: close their visit without guessing theft
        keep = []
        for j, lost_at in self.lost:
            if now_s - lost_at > LOST_JOURNEY_S:
                self._finish_visit(j, now, frame, seen_leaving=False)
            else:
                keep.append((j, lost_at))
        self.lost = keep

        counter = to_px(self.layout["counter"], w, h)
        staff = to_px(self.layout["staff_zone"], w, h)
        slots = [(s, to_px(s["box"], w, h)) for s in self.layout.get("shelves", [])]
        at_counter, in_staff = 0, []
        presence = self.count_mode == "presence"
        for box, tid in zip(people, self._det_tracks):
            ft = feet(box)
            j = self.journeys.get(tid)
            tr = self.tracks.get(tid)
            at_ctr = presence or inside(ft, counter)        # a counter camera: everyone it sees is at the counter
            if at_ctr:
                at_counter += 1
            centre = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
            if (inside(centre, staff) if presence else inside(ft, staff)) and (after_hours or (j is not None and not presence)):
                in_staff.append(box)
            if j is None:
                continue
            dt = min(1.0, max(0.0, now_s - getattr(tr, "_jt", now_s))) if tr is not None else 0.0
            if tr is not None:
                tr._jt = now_s
            if at_ctr:
                j.counter_s += dt
                if not j.reached_counter and j.counter_s >= COUNTER_SECONDS:
                    j.reached_counter = True
                    self._event("counter", "Customer at the billing counter")
            for s, corners in slots:
                sw = corners[2] - corners[0]
                if corners[0] - 0.2 * sw <= ft[0] <= corners[2] + 0.2 * sw and ft[1] >= corners[3] \
                        and overlap_frac(box, corners) >= 0.3:
                    j.near_since[s["id"]] = now_s
                    j.near_s[s["id"]] = j.near_s.get(s["id"], 0.0) + dt
        return at_counter, in_staff

    def _update_shelves(self, frame: np.ndarray, people, now: datetime, now_s: float) -> None:
        if not self.shelves.slots:
            return
        for slot, delta, cans in self.shelves.update(frame, people):
            sid, pid = slot["id"], slot.get("product_id")
            label = slot.get("label") or slot.get("name", sid)
            if delta < 0:
                tid = credit_pick(self.journeys, sid, now_s)
                if tid is not None:
                    j = self.journeys[tid]
                    j.picks[sid] = j.picks.get(sid, 0) + (-delta)
                if self.log_to_db:
                    log_shelf_event(self.data_source, sid, pid, "pick", -delta, at=now.replace(tzinfo=None))
                self._event("pick", f"{-delta} × {label} taken from {slot.get('name', sid)}"
                            + ("" if tid is not None else " (no customer seen there)"))
            else:
                back = 0
                tid = credit_put_back(self.journeys, sid, now_s)
                if tid is not None:
                    j = self.journeys[tid]
                    back = min(delta, j.picks.get(sid, 0))
                    j.picks[sid] -= back
                    j.put_back += back
                    if self.log_to_db:
                        log_shelf_event(self.data_source, sid, pid, "put_back", back, at=now.replace(tzinfo=None))
                    self._event("put_back", f"{back} × {label} put back on {slot.get('name', sid)}")
                if delta - back > 0:
                    if self.log_to_db:
                        log_shelf_event(self.data_source, sid, pid, "restock", delta - back, at=now.replace(tzinfo=None))
                    self._event("restock", f"{slot.get('name', sid)} refilled: +{delta - back} {label}")
        counts = self.shelves.counts()
        if self.log_to_db:
            for s in self.shelves.slots:
                n = counts.get(s["id"])
                if n is not None and self._saved_shelves.get(s["id"]) != n:
                    set_shelf_status(self.data_source, s["id"], s.get("name", s["id"]), s.get("product_id"), n,
                                     int(s.get("capacity", 4)), at=now.replace(tzinfo=None))
                    self._saved_shelves[s["id"]] = n

    def _motion_fraction(self, frame: np.ndarray) -> float:
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        fg = self.bg.apply(small)
        fg = cv2.morphologyEx((fg == 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return float(fg.mean())

    def _save_snapshot(self, frame: np.ndarray, kind: str, at: datetime) -> str:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOT_DIR / f"{at.strftime('%Y%m%d_%H%M%S')}_{kind.lower()}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path.relative_to(DATA_DIR.parent))

    def _maybe_alert(self, kind: str, zone: str, details: str, frame: np.ndarray, at: datetime,
                     key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        now_s = time.time()
        key = key or kind
        if now_s - self.last_alert_at.get(key, 0.0) < SECURITY_ALERT_COOLDOWN_SECONDS:
            return None
        self.last_alert_at[key] = now_s
        alert = {"type": kind, "zone": zone, "severity": "CRITICAL", "details": details, "saved": False}
        if self.log_to_db:
            photo = self._save_snapshot(frame, kind, at)
            alert["details"] = f"{details} Photo: {photo}"
            log_security_event(kind, zone, "CRITICAL", alert["details"], at=at.replace(tzinfo=None),
                               source=self.data_source)
            alert["saved"] = True
        self.last_alert = alert
        self._event("alert", details)
        return alert

    # ---------------------------------------------------------------- main step
    def process_frame(self, frame: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Read (or take) one frame, analyse it, draw overlays.
        Returns (annotated BGR frame or None if no video, metrics dict).
        """
        now = clock.now()
        if frame is None:
            frame = self.read_frame()
        if frame is None:
            return None, {"connected": False, "status": self.status_message, "current_people": 0,
                          "entered": self.total_entered, "exited": self.total_exited,
                          "is_after_hours": not clock.is_store_open(now), "security_alert": None,
                          "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")}
        frame = frame.copy()
        raw = frame.copy()                      # shelves are read from the clean picture
        self.latest_raw = raw
        h, w = frame.shape[:2]
        self._frame_w = w
        self.frames_seen += 1
        line_y = int(self.door_line_y * h)
        after_hours = not clock.is_store_open(now)
        now_s = time.time()
        presence = self.count_mode == "presence"

        people = self.detect_people(frame)
        faces = self.detect_faces(frame, people)
        if self.count_mode == "presence":
            people, faces = people + faces, []      # a face is a person too (webcams rarely see whole bodies)
        entered, exited = self._update_tracks(people, line_y, now_s, h)
        if entered or exited:
            self.total_entered += entered
            self.total_exited += exited
            if self.log_to_db:
                log_footfall(count_in=entered, count_out=exited, at=now.replace(tzinfo=None), source=self.data_source)
        self.current_occupancy = max(0, self.total_entered - self.total_exited)

        at_counter, in_staff = self._update_journeys(raw, people, now, now_s, after_hours)
        self._update_shelves(raw, people + faces, now, now_s)

        alert = None
        motion = self._motion_fraction(frame) if self.source_type != "demo" else 0.0
        warming_up = self.frames_seen < 15  # background model needs a few frames
        if after_hours and not warming_up and (motion > 0.015 or people or faces):
            alert = self._maybe_alert(
                "AFTER_HOURS_MOTION", "Main store area",
                f"Movement at {clock.fmt_time(now)} while the shop is closed.", raw, now) or alert
        if in_staff:
            alert = self._maybe_alert(
                "RESTRICTED_ZONE_ENTRY", "Store room (staff only)",
                f"A customer went into the staff-only area at {clock.fmt_time(now)}.", raw, now) or alert
        if self.last_alert and alert is None and self.last_alert.get("_shown", 0) < 1:
            alert = self.last_alert              # walk-out alerts raised while closing a visit
            self.last_alert["_shown"] = 1

        # ---- overlays
        zx1, zy1, zx2, zy2 = to_px(self.layout["staff_zone"], w, h)
        cx1, cy1, cx2, cy2 = to_px(self.layout["counter"], w, h)
        door = self.layout["door"]
        dx1, dx2 = int(door.get("x1", 0.0) * w), int(door.get("x2", 1.0) * w)
        if not presence:            # a counter camera has no door line, counter box or staff area to show
            cv2.line(frame, (dx1, line_y), (dx2, line_y), YELLOW, 2)
            _tag(frame, "Door line", dx1 + 4, line_y - 6, YELLOW)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), BLUE, 2)
            _tag(frame, "Billing counter", cx1 + 2, cy1 + 18, BLUE)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), RED, 2)
            _tag(frame, "Staff only", zx1 + 2, zy1 + 18, RED)
        counts = self.shelves.counts()
        for s_ in self.shelves.slots:
            sx1, sy1, sx2, sy2 = to_px(s_["box"], w, h)
            n, cap = counts.get(s_["id"]), int(s_.get("capacity", 4))
            low = n is not None and n <= max(1, cap // 4)
            col = RED if low else (WHITE if n is None else GREEN)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), col, 1)
            tag = f"{s_['id']} {'?' if n is None else n}/{cap}"
            cv2.rectangle(frame, (sx1, sy2 - 16), (sx1 + 8 * len(tag), sy2), DARK, -1)
            cv2.putText(frame, tag, (sx1 + 3, sy2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

        box_tags = []
        for (x, y, bw, bh), tid in zip(people, self._det_tracks):
            j = self.journeys.get(tid)
            tr = self.tracks.get(tid)
            if j is None:
                colour, tag = (190, 190, 190), ("Person · checking…" if presence else "Person")
            elif j.net_picked and not j.reached_counter:
                colour, tag = RED, f"Customer · {j.net_picked} can{'s' if j.net_picked != 1 else ''}"
            elif presence:
                secs = now_s - (tr.first_seen if tr else now_s)
                colour, tag = GREEN, f"Customer · {secs / 60:.0f} min" if secs >= 90 else f"Customer · {secs:.0f} s"
            elif j.reached_counter:
                colour, tag = BLUE, "Customer · at counter"
            else:
                colour, tag = GREEN, "Customer"
            if any(b == (x, y, bw, bh) for b in in_staff):
                colour = RED
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)
            _tag(frame, tag, x, y - 4, colour)
            box_tags.append(tag)

        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), GREEN, 2)
            cv2.putText(frame, "Person", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)

        label = SOURCE_LABELS[self.source_type] + (" (drawn example)" if self.source_type == "demo" else "")
        cv2.putText(frame, f"{label} | {now.strftime('%H:%M:%S')}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        return frame, {
            "connected": True,
            "status": self.status_message,
            "current_people": len(people) + len(faces),
            "entered": self.total_entered,
            "exited": self.total_exited,
            "occupancy": self.current_occupancy,
            "inside_now": len(self.journeys),
            "count_mode": self.count_mode,
            "at_counter": at_counter,
            "shelves": counts,
            "events": list(self.events[:12]),
            "is_after_hours": after_hours,
            "security_alert": alert,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            # Boxes as fractions of the frame, for drawing over a live browser video
            "boxes": [{"x": x / w, "y": y / h, "w": bw / w, "h": bh / h, "label": t,
                       "zone": any(b == (x, y, bw, bh) for b in in_staff), "kind": "person"}
                      for (x, y, bw, bh), t in zip(people, box_tags)]
                     + [{"x": x / w, "y": y / h, "w": fw / w, "h": fh / h, "zone": False, "kind": "face"}
                        for (x, y, fw, fh) in faces],
        }
