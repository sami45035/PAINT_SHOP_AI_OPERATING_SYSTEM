"""
Draws the three sample rooms used in Colour Studio when no customer photo is
uploaded. They are illustrations, not photos, but they have what the wall
tools need to show off: three walls at different angles, soft window light,
corner shadows, a ceiling, a wooden floor and furniture in front of the wall.

Run:  python data/make_sample_rooms.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SAMPLE_ROOMS_DIR  # noqa: E402

W, H = 1200, 800
RNG = np.random.default_rng(7)


def poly(points):
    return np.array([[int(x * W), int(y * H)] for x, y in points], dtype=np.int32)


def mask_of(points):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly(points)], 255)
    return m > 0


def paint(img, mask, colour, light=None):
    c = np.array(colour, np.float32)
    layer = np.broadcast_to(c, img.shape).astype(np.float32)
    if light is not None:
        layer = layer * light[..., None]
    img[mask] = layer[mask]


def base_room(wall_rgb, window_side="right"):
    img = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bx0, by0, bx1, by1 = 0.22, 0.16, 0.78, 0.74

    ceiling = mask_of([(0, 0), (1, 0), (bx1, by0), (bx0, by0)])
    left = mask_of([(0, 0), (bx0, by0), (bx0, by1), (0, 1.0)])
    right = mask_of([(1, 0), (bx1, by0), (bx1, by1), (1, 1.0)])
    back = mask_of([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)])
    floor = mask_of([(0, 1.0), (bx0, by1), (bx1, by1), (1, 1.0)])

    # Window light: brighter toward the window wall, darker into corners
    wx = W * (0.95 if window_side == "right" else 0.05)
    dist = np.hypot(xx - wx, (yy - H * 0.35) * 0.8) / np.hypot(W, H)
    window_glow = 1.08 - 0.32 * dist
    corner = np.minimum.reduce([
        np.abs(xx - bx0 * W), np.abs(xx - bx1 * W), np.abs(yy - by0 * H) * 1.5,
    ]) / (0.08 * W)
    ao = 0.86 + 0.14 * np.clip(corner, 0, 1)
    noise = 1.0 + RNG.normal(0, 0.012, (H, W)).astype(np.float32)
    light = window_glow * ao * noise

    paint(img, back, wall_rgb, light * 1.00)
    paint(img, left, wall_rgb, light * (0.80 if window_side == "right" else 0.95))
    paint(img, right, wall_rgb, light * (0.93 if window_side == "right" else 0.80))
    paint(img, ceiling, (246, 244, 239), np.clip(light * 1.02, 0, 1.05))

    # Wooden floor: planks with grain
    planks = (np.sin(xx / 7.0 + (yy / 23.0) ** 1.3) * 6 + ((xx // 70) % 3) * 7).astype(np.float32)
    floor_light = (0.85 + 0.25 * (yy / H)) * noise
    wood = np.stack([170 + planks, 122 + planks * 0.8, 82 + planks * 0.5], axis=-1) * floor_light[..., None]
    img[floor] = wood[floor]

    # Skirting along the back wall
    cv2.line(img, (int(bx0 * W), int(by1 * H) - 3), (int(bx1 * W), int(by1 * H) - 3), (238, 236, 230), 6)
    return img, light


def window(img, side="right"):
    if side == "right":
        pts = [(0.86, 0.26), (0.95, 0.20), (0.95, 0.56), (0.86, 0.57)]
    else:
        pts = [(0.05, 0.20), (0.14, 0.26), (0.14, 0.57), (0.05, 0.56)]
    cv2.fillPoly(img, [poly(pts)], (250, 250, 248))
    inner = [(x + (0.01 if i in (0, 3) else -0.01), y + (0.015 if i in (0, 1) else -0.015)) for i, (x, y) in enumerate(pts)]
    cv2.fillPoly(img, [poly(inner)], (196, 222, 240))
    mx = sum(p[0] for p in pts) / 4
    cv2.line(img, (int(mx * W), int(0.23 * H)), (int(mx * W), int(0.56 * H)), (250, 250, 248), 5)


def door(img, side="left"):
    pts = [(0.05, 0.27), (0.15, 0.24), (0.15, 0.82), (0.05, 0.92)] if side == "left" else \
          [(0.85, 0.24), (0.95, 0.27), (0.95, 0.92), (0.85, 0.82)]
    cv2.fillPoly(img, [poly(pts)], (120, 82, 56))
    knob = (int(0.135 * W), int(0.56 * H)) if side == "left" else (int(0.865 * W), int(0.56 * H))
    cv2.circle(img, knob, 6, (210, 190, 120), -1, cv2.LINE_AA)


def frame(img, x, y, w, h, art=(60, 72, 70)):
    cv2.rectangle(img, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (45, 45, 45), -1)
    cv2.rectangle(img, (int(x * W) + 8, int(y * H) + 8), (int((x + w) * W) - 8, int((y + h) * H) - 8), (248, 247, 242), -1)
    cx, cy = int((x + w / 2) * W), int((y + h / 2) * H)
    cv2.ellipse(img, (cx, cy), (int(w * W * 0.18), int(h * H * 0.3)), 35, 0, 360, art, -1, cv2.LINE_AA)


def bedroom():
    img, _ = base_room((226, 214, 196), "right")
    window(img, "right")
    door(img, "left")
    frame(img, 0.45, 0.24, 0.10, 0.17)
    cv2.rectangle(img, (int(0.33 * W), int(0.52 * H)), (int(0.67 * W), int(0.70 * H)), (150, 104, 70), -1)  # headboard
    cv2.fillPoly(img, [poly([(0.30, 0.66), (0.70, 0.66), (0.78, 0.94), (0.22, 0.94)])], (246, 244, 238))      # mattress
    cv2.fillPoly(img, [poly([(0.26, 0.80), (0.74, 0.80), (0.755, 0.85), (0.245, 0.85)])], (150, 150, 150))   # blanket band
    for x in (0.36, 0.53):
        cv2.rectangle(img, (int(x * W), int(0.61 * H)), (int((x + 0.11) * W), int(0.68 * H)), (234, 231, 224), -1)
    return img


def living_room():
    img, _ = base_room((212, 214, 212), "left")
    window(img, "left")
    door(img, "right")
    frame(img, 0.36, 0.25, 0.12, 0.15, art=(176, 120, 72))
    frame(img, 0.52, 0.25, 0.12, 0.15, art=(70, 96, 120))
    cv2.fillPoly(img, [poly([(0.28, 0.80), (0.72, 0.80), (0.80, 0.96), (0.20, 0.96)])], (190, 170, 140))      # rug
    cv2.rectangle(img, (int(0.30 * W), int(0.55 * H)), (int(0.70 * W), int(0.66 * H)), (52, 66, 92), -1)      # sofa back
    cv2.rectangle(img, (int(0.28 * W), int(0.64 * H)), (int(0.72 * W), int(0.78 * H)), (60, 76, 104), -1)     # sofa seat
    for x in (0.34, 0.58):
        cv2.rectangle(img, (int(x * W), int(0.58 * H)), (int((x + 0.08) * W), int(0.65 * H)), (214, 168, 92), -1)
    return img


def dining_room():
    img, _ = base_room((236, 228, 210), "right")
    window(img, "right")
    frame(img, 0.42, 0.22, 0.16, 0.2, art=(150, 60, 50))
    cv2.fillPoly(img, [poly([(0.32, 0.70), (0.68, 0.70), (0.72, 0.78), (0.28, 0.78)])], (128, 88, 58))       # table top
    for x in (0.31, 0.67):
        cv2.rectangle(img, (int(x * W), int(0.78 * H)), (int((x + 0.015) * W), int(0.93 * H)), (110, 74, 48), -1)
    for x in (0.25, 0.70):
        cv2.rectangle(img, (int(x * W), int(0.60 * H)), (int((x + 0.05) * W), int(0.86 * H)), (92, 62, 40), -1)  # chairs
    return img


def main():
    out = {"master_bedroom.jpg": bedroom(), "modern_living_room.jpg": living_room(), "cozy_dining_room.jpg": dining_room()}
    for name, img in out.items():
        img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
        cv2.imwrite(str(SAMPLE_ROOMS_DIR / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", SAMPLE_ROOMS_DIR / name)


if __name__ == "__main__":
    main()
