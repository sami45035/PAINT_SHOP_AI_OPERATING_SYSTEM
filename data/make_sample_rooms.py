"""
Draws the three sample rooms used in Colour Studio when no customer photo is
uploaded. They are illustrations, not photos, but they have what the wall
tools need to show off: walls at different angles, soft window light, warm
light pools from wall lamps, corner shadows, a ceiling, a wooden floor and
furniture in front of the wall.

Each room is drawn together with a label map (what every pixel is: wall,
ceiling, bed, window…) so tests can measure how accurately walls are found.

Run:  python data/make_sample_rooms.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SAMPLE_ROOMS_DIR  # noqa: E402

W, H = 1200, 800

# ADE20K class ids (same numbering as the wall model)
WALL, FLOOR, CEILING, BED, WINDOW, CURTAIN, DOOR, TABLE, CHAIR, PAINTING, SOFA, RUG, LAMP, CUSHION, PILLOW = \
    0, 3, 5, 7, 8, 18, 14, 15, 19, 22, 23, 28, 36, 39, 57
ROOMS = {"master_bedroom.jpg": "bedroom", "modern_living_room.jpg": "living_room", "cozy_dining_room.jpg": "dining_room"}


class Canvas:
    """Draws on the picture and the label map at the same time."""

    def __init__(self, seed: int):
        self.img = np.zeros((H, W, 3), np.float32)
        self.lab = np.full((H, W), WALL, np.int32)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def pts(points):
        return np.array([[int(x * W), int(y * H)] for x, y in points], dtype=np.int32)

    def poly(self, points, colour, cls):
        p = self.pts(points)
        cv2.fillPoly(self.img, [p], colour)
        cv2.fillPoly(self.lab, [p], int(cls))

    def rect(self, x0, y0, x1, y1, colour, cls):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], colour, cls)

    def mask(self, points):
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [self.pts(points)], 255)
        return m > 0


BX0, BY0, BX1, BY1 = 0.22, 0.16, 0.78, 0.74


def base_room(c: Canvas, wall_rgb, window_side="right", lamps=()):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ceiling = c.mask([(0, 0), (1, 0), (BX1, BY0), (BX0, BY0)])
    left = c.mask([(0, 0), (BX0, BY0), (BX0, BY1), (0, 1.0)])
    right = c.mask([(1, 0), (BX1, BY0), (BX1, BY1), (1, 1.0)])
    back = c.mask([(BX0, BY0), (BX1, BY0), (BX1, BY1), (BX0, BY1)])
    floor = c.mask([(0, 1.0), (BX0, BY1), (BX1, BY1), (1, 1.0)])

    # Window light: brighter toward the window wall, darker into corners
    wx = W * (0.95 if window_side == "right" else 0.05)
    dist = np.hypot(xx - wx, (yy - H * 0.35) * 0.8) / np.hypot(W, H)
    light = 1.08 - 0.32 * dist
    corner = np.minimum.reduce([np.abs(xx - BX0 * W), np.abs(xx - BX1 * W), np.abs(yy - BY0 * H) * 1.5]) / (0.08 * W)
    light *= 0.86 + 0.14 * np.clip(corner, 0, 1)
    # Warm wall-lamp light pools near the ceiling (these fool wall models into saying "ceiling")
    for lx, ly, r in lamps:
        d = np.hypot(xx - lx * W, (yy - ly * H) * 1.3) / (r * W)
        light += 0.95 * np.exp(-d ** 2)
    light *= 1.0 + c.rng.normal(0, 0.012, (H, W)).astype(np.float32)

    def paint(mask, colour, gain):
        layer = np.array(colour, np.float32)[None, None, :] * (light * gain)[..., None]
        c.img[mask] = layer[mask]

    paint(back, wall_rgb, 1.00)
    paint(left, wall_rgb, 0.80 if window_side == "right" else 0.95)
    paint(right, wall_rgb, 0.93 if window_side == "right" else 0.80)
    paint(ceiling, (246, 244, 239), 1.02)
    c.lab[ceiling] = CEILING

    planks = (np.sin(xx / 7.0 + (yy / 23.0) ** 1.3) * 6 + ((xx // 70) % 3) * 7).astype(np.float32)
    floor_light = 0.85 + 0.25 * (yy / H)
    wood = np.stack([170 + planks, 122 + planks * 0.8, 82 + planks * 0.5], axis=-1) * floor_light[..., None]
    c.img[floor] = wood[floor]
    c.lab[floor] = FLOOR
    # Skirting along the back wall
    c.rect(BX0, BY1 - 0.008, BX1, BY1, (238, 236, 230), WALL)


def window(c: Canvas, side="right"):
    pts = [(0.86, 0.26), (0.95, 0.20), (0.95, 0.56), (0.86, 0.57)] if side == "right" else \
          [(0.05, 0.20), (0.14, 0.26), (0.14, 0.57), (0.05, 0.56)]
    c.poly(pts, (250, 250, 248), WINDOW)
    inner = [(x + (0.01 if i in (0, 3) else -0.01), y + (0.015 if i in (0, 1) else -0.015)) for i, (x, y) in enumerate(pts)]
    c.poly(inner, (196, 222, 240), WINDOW)
    mx = sum(p[0] for p in pts) / 4
    cv2.line(c.img, (int(mx * W), int(0.23 * H)), (int(mx * W), int(0.56 * H)), (250, 250, 248), 5)


def door(c: Canvas, side="left"):
    pts = [(0.05, 0.27), (0.15, 0.24), (0.15, 0.82), (0.05, 0.92)] if side == "left" else \
          [(0.85, 0.24), (0.95, 0.27), (0.95, 0.92), (0.85, 0.82)]
    c.poly(pts, (120, 82, 56), DOOR)
    knob = (int(0.135 * W), int(0.56 * H)) if side == "left" else (int(0.865 * W), int(0.56 * H))
    cv2.circle(c.img, knob, 6, (210, 190, 120), -1, cv2.LINE_AA)


def frame(c: Canvas, x, y, w, h, art=(60, 72, 70)):
    c.rect(x, y, x + w, y + h, (45, 45, 45), PAINTING)
    cv2.rectangle(c.img, (int(x * W) + 8, int(y * H) + 8), (int((x + w) * W) - 8, int((y + h) * H) - 8), (248, 247, 242), -1)
    cx, cy = int((x + w / 2) * W), int((y + h / 2) * H)
    cv2.ellipse(c.img, (cx, cy), (int(w * W * 0.18), int(h * H * 0.3)), 35, 0, 360, art, -1, cv2.LINE_AA)


def sconce(c: Canvas, x, y):
    """Small wall lamp (the light pool around it is drawn by base_room)."""
    c.rect(x - 0.012, y - 0.01, x + 0.012, y + 0.035, (236, 226, 204), LAMP)
    c.rect(x - 0.004, y + 0.035, x + 0.004, y + 0.05, (90, 80, 70), LAMP)


def bedroom():
    c = Canvas(7)
    base_room(c, (226, 214, 196), "right", lamps=[(0.30, 0.20, 0.10), (0.70, 0.20, 0.10), (0.84, 0.10, 0.08)])
    window(c, "right")
    door(c, "left")
    frame(c, 0.45, 0.24, 0.10, 0.17)
    sconce(c, 0.30, 0.30)
    sconce(c, 0.70, 0.30)
    c.rect(0.33, 0.52, 0.67, 0.70, (236, 232, 222), BED)                  # pale panelled headboard
    for i in range(1, 6):
        yv = int((0.52 + i * 0.03) * H)
        cv2.line(c.img, (int(0.33 * W), yv), (int(0.67 * W), yv), (214, 208, 196), 2)
    c.poly([(0.30, 0.66), (0.70, 0.66), (0.78, 0.94), (0.22, 0.94)], (246, 244, 238), BED)
    c.poly([(0.26, 0.80), (0.74, 0.80), (0.755, 0.85), (0.245, 0.85)], (34, 110, 84), BED)   # green throw
    for x in (0.36, 0.53):
        c.rect(x, 0.61, x + 0.11, 0.68, (40, 120, 92), PILLOW)
    return c


def living_room():
    c = Canvas(11)
    base_room(c, (212, 214, 212), "left", lamps=[(0.50, 0.19, 0.12)])
    window(c, "left")
    door(c, "right")
    frame(c, 0.36, 0.25, 0.12, 0.15, art=(176, 120, 72))
    frame(c, 0.52, 0.25, 0.12, 0.15, art=(70, 96, 120))
    c.poly([(0.28, 0.80), (0.72, 0.80), (0.80, 0.96), (0.20, 0.96)], (190, 170, 140), RUG)
    c.rect(0.30, 0.55, 0.70, 0.66, (52, 66, 92), SOFA)
    c.rect(0.28, 0.64, 0.72, 0.78, (60, 76, 104), SOFA)
    for x in (0.34, 0.58):
        c.rect(x, 0.58, x + 0.08, 0.65, (214, 168, 92), CUSHION)
    return c


def dining_room():
    c = Canvas(13)
    base_room(c, (236, 228, 210), "right", lamps=[(0.36, 0.20, 0.09), (0.64, 0.20, 0.09)])
    window(c, "right")
    frame(c, 0.42, 0.22, 0.16, 0.2, art=(150, 60, 50))
    c.poly([(0.32, 0.70), (0.68, 0.70), (0.72, 0.78), (0.28, 0.78)], (128, 88, 58), TABLE)
    for x in (0.31, 0.67):
        c.rect(x, 0.78, x + 0.015, 0.93, (110, 74, 48), TABLE)
    for x in (0.25, 0.70):
        c.rect(x, 0.60, x + 0.05, 0.86, (92, 62, 40), CHAIR)
    return c


DRAW = {"bedroom": bedroom, "living_room": living_room, "dining_room": dining_room}


def render(file_name: str):
    """(RGB uint8 image, int label map) for one sample room."""
    c = DRAW[ROOMS[file_name]]()
    img = cv2.GaussianBlur(np.clip(c.img, 0, 255).astype(np.uint8), (3, 3), 0)
    return img, c.lab


def main():
    for name in ROOMS:
        img, _ = render(name)
        cv2.imwrite(str(SAMPLE_ROOMS_DIR / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", SAMPLE_ROOMS_DIR / name)


if __name__ == "__main__":
    main()
