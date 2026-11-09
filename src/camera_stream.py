"""
Store Camera (Layer 5): door counting and after-hours guard.

Sources
- "webcam": laptop / USB camera (works when the app runs on that laptop)
- "esp32":  ESP32-CAM MJPEG stream on the shop Wi-Fi
- "demo":   a drawn scene for showing the feature. Clearly labelled, its
            numbers are examples, and it NEVER writes to the database.

What it does with real video
- Finds people (OpenCV HOG detector + non-maximum suppression).
- Follows each person between frames and counts them crossing the door line
  (downward = coming in, upward = going out). Counts go into footfall_logs.
- After closing time, any noticeable movement raises AFTER_HOURS_MOTION.
- A person inside the staff-only zone raises RESTRICTED_ZONE_ENTRY.
- Each alert type is saved at most once per SECURITY_ALERT_COOLDOWN_SECONDS,
  with a snapshot photo.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
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
from src.database import log_footfall, log_security_event

SNAPSHOT_DIR = DATA_DIR / "security_snapshots"
SOURCE_LABELS = {"webcam": "Laptop webcam", "esp32": "ESP32-CAM", "demo": "Demo feed"}

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


class StoreVisionEngine:
    def __init__(self, source_type: str = "webcam", esp32_url: str = ESP32_DEFAULT_STREAM_URL,
                 log_to_db: Optional[bool] = None):
        if source_type == "synthetic":  # old name
            source_type = "demo"
        self.source_type = source_type
        self.esp32_url = esp32_url or ESP32_DEFAULT_STREAM_URL
        self.log_to_db = (source_type != "demo") if log_to_db is None else (log_to_db and source_type != "demo")
        self.cap: Optional[cv2.VideoCapture] = None
        self.connected = False
        self.door_line_y = DOOR_LINE_Y                  # adjustable from the Store Camera screen
        self.zone_box = list(RESTRICTED_ZONE_BOX)
        self.status_message = ""

        # Session counters (the database keeps the day's totals for real sources)
        self.total_entered = 0
        self.total_exited = 0
        self.current_occupancy = 0

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.last_alert_at: Dict[str, float] = {}
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
        self.frames_seen = 0

        # Faces: so someone close to the camera (at the counter) is still seen
        self.face = None
        try:
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if path.exists():
                self.face = cv2.CascadeClassifier(str(path))
                if self.face.empty():
                    self.face = None
        except Exception:
            self.face = None

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

        # Demo walker state
        self._demo_t0 = time.time()

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
    def _demo_person(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """A person walking in through the door, pausing, and walking out — on a 16 s loop."""
        t = (time.time() - self._demo_t0) % 16.0
        door_y = int(self.door_line_y * h)
        if t < 6:      # walking in (top of frame toward the camera)
            y = int(h * 0.30 + (door_y + 60 - h * 0.30) * (t / 6))
        elif t < 9:    # browsing
            y = door_y + 60
        else:          # walking back out
            y = int(door_y + 60 - (door_y + 60 - h * 0.30) * ((t - 9) / 7))
        x = int(w * 0.50 + 30 * math.sin(t))
        bw, bh = 56, 150
        return x - bw // 2, y - bh, bw, bh

    def generate_synthetic_store_frame(self) -> np.ndarray:
        h, w = 480, 640
        frame = np.full((h, w, 3), (60, 49, 38), dtype=np.uint8)          # dark wall
        cv2.rectangle(frame, (0, int(h * 0.66)), (w, h), (76, 64, 51), -1)  # floor
        cv2.rectangle(frame, (30, 90), (140, 390), (94, 82, 70), -1)        # paint shelf
        for sy in (150, 210, 270, 330):
            cv2.line(frame, (30, sy), (140, sy), (131, 119, 107), 3)
            for i, col in enumerate([(50, 90, 200), (70, 160, 60), (180, 140, 50)]):
                cv2.circle(frame, (55 + i * 30, sy - 18), 12, col, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (250, 95), (400, 380), (131, 119, 107), 2)     # doorway
        x, y, bw, bh = self._demo_person(w, h)
        cx = x + bw // 2
        cv2.circle(frame, (cx, y + 18), 16, (150, 170, 190), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx - 20, y + 36), (cx + 20, y + 100), (120, 70, 40), -1)
        cv2.line(frame, (cx - 9, y + 100), (cx - 13, y + bh), (40, 40, 50), 6)
        cv2.line(frame, (cx + 9, y + 100), (cx + 13, y + bh), (40, 40, 50), 6)
        return frame

    # ---------------------------------------------------------------- detection
    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Whole people (standing/walking). Used for door counting and the staff-only zone."""
        if self.source_type == "demo":
            h, w = frame.shape[:2]
            return [self._demo_person(w, h)]
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
        faces = self.face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(min_side, min_side))
        out = []
        for (x, y, fw, fh) in (faces if len(faces) else []):
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
                tr = Track(self.next_track_id, c, (x, y, bw, bh), now_s, history=[c[1]])
                self.tracks[tr.id] = tr
                self.next_track_id += 1
            else:
                unmatched.discard(best)
                tr = self.tracks[best]
                tr.centroid, tr.box, tr.last_seen = c, (x, y, bw, bh), now_s
                tr.frames += 1
                tr.history = (tr.history + [c[1]])[-20:]
            if not feet_visible:
                continue
            side = "out" if c[1] < line_y - margin else ("in" if c[1] > line_y + margin else None)
            if side is None:
                continue
            if tr.side == "out" and side == "in" and tr.frames >= 2 and not tr.counted_in:
                entered += 1
                tr.counted_in = True
            elif tr.side == "in" and side == "out" and tr.frames >= 2 and not tr.counted_out:
                exited += 1
                tr.counted_out = True
            tr.side = side
        stale = 4.0 if self.source_type == "demo" else 1.5
        for tid in [t for t, tr in self.tracks.items() if now_s - tr.last_seen > stale]:
            del self.tracks[tid]
        return entered, exited

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

    def _maybe_alert(self, kind: str, zone: str, details: str, frame: np.ndarray, at: datetime) -> Optional[Dict[str, Any]]:
        now_s = time.time()
        if now_s - self.last_alert_at.get(kind, 0.0) < SECURITY_ALERT_COOLDOWN_SECONDS:
            return None
        self.last_alert_at[kind] = now_s
        alert = {"type": kind, "zone": zone, "severity": "CRITICAL", "details": details, "saved": False}
        if self.log_to_db:
            photo = self._save_snapshot(frame, kind, at)
            alert["details"] = f"{details} Photo: {photo}"
            log_security_event(kind, zone, "CRITICAL", alert["details"], at=at.replace(tzinfo=None))
            alert["saved"] = True
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
        h, w = frame.shape[:2]
        self.frames_seen += 1
        line_y = int(self.door_line_y * h)
        after_hours = not clock.is_store_open(now)

        people = self.detect_people(frame)
        faces = self.detect_faces(frame, people)
        entered, exited = self._update_tracks(people, line_y, time.time(), h)
        if entered or exited:
            self.total_entered += entered
            self.total_exited += exited
            if self.log_to_db:
                log_footfall(count_in=entered, count_out=exited, at=now.replace(tzinfo=None))
        self.current_occupancy = max(0, self.total_entered - self.total_exited)

        # Zones
        zx1, zy1 = int(w * self.zone_box[0]), int(h * self.zone_box[1])
        zx2, zy2 = int(w * self.zone_box[2]), int(h * self.zone_box[3])

        alert = None
        motion = self._motion_fraction(frame) if self.source_type != "demo" else 0.0
        warming_up = self.frames_seen < 15  # background model needs a few frames
        if after_hours and not warming_up and (motion > 0.015 or people or faces):
            alert = self._maybe_alert(
                "AFTER_HOURS_MOTION", "Main store area",
                f"Movement at {clock.fmt_time(now)} while the shop is closed.", frame, now) or alert
        for (x, y, bw, bh) in people:
            cx, cy = x + bw // 2, y + bh // 2
            in_zone = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
            colour = RED if in_zone else GREEN
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)
            if in_zone:
                alert = self._maybe_alert(
                    "RESTRICTED_ZONE_ENTRY", "Storage room (staff only)",
                    f"Someone entered the staff-only zone at {clock.fmt_time(now)}.", frame, now) or alert

        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), GREEN, 2)
            cv2.putText(frame, "Person", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)

        # Overlays
        cv2.line(frame, (0, line_y), (w, line_y), YELLOW, 2)
        cv2.putText(frame, "Door line", (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), RED, 2)
        cv2.putText(frame, "Staff only", (zx1 + 6, zy1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
        label = SOURCE_LABELS[self.source_type] + (" (not real)" if self.source_type == "demo" else "")
        cv2.putText(frame, f"{label} | {now.strftime('%H:%M:%S')}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        return frame, {
            "connected": True,
            "status": self.status_message,
            "current_people": len(people) + len(faces),
            "entered": self.total_entered,
            "exited": self.total_exited,
            "occupancy": self.current_occupancy,
            "is_after_hours": after_hours,
            "security_alert": alert,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
