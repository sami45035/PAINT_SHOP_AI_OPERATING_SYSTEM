"""
Fake camera videos for the browser end-to-end test (Chromium plays them as the webcam).
  shop.mjpeg     the laptop at the counter: customer A buys, staff in uniform, customer B asks, a passer-by
  labels.mjpeg   the phone at the till: two cans' QR labels held up one after the other
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.people_scene import background, scene  # noqa: E402

FPS = 30                      # Chromium plays MJPEG files at 30 frames a second
ORANGE, RED, GREEN, NAVY = (0, 120, 245), (40, 40, 200), (60, 160, 60), (90, 40, 30)

SHOP_CAST = [   # (name, shirt, [(second, x, feet y, height)]) — fractions of the picture
    ("customer_a", RED,    [(6, .25, .93, .62), (8, .38, .93, .64), (78, .38, .93, .64), (80, .25, .93, .62)]),
    ("staff",      ORANGE, [(15, .75, .92, .60), (17, .62, .92, .62), (148, .62, .92, .62), (150, .75, .92, .60)]),
    ("customer_b", GREEN,  [(92, .20, .93, .62), (94, .30, .93, .63), (133, .30, .93, .63), (135, .20, .93, .62)]),
    ("passer_by",  NAVY,   [(155, .05, .93, .60), (155.8, .25, .93, .60)]),
]
SHOP_SECONDS = 175
LABELS = [(1.5, 4.5, "APX:SKU-AS-1001", "Royale Blue 1 L"), (7.0, 10.0, "APX:SKU-NE-1070", "Honey Dew 4 L")]
LABEL_SECONDS = 90


def shop_video(path: Path, w: int = 640, h: int = 360) -> None:
    bg = background(w, h)
    with open(path, "wb") as f:
        for i in range(SHOP_SECONDS * FPS):
            frame, _ = scene(i / FPS, SHOP_CAST, w, h, bg=bg)
            f.write(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())


def label_video(path: Path, w: int = 640, h: int = 480) -> None:
    from src.billing import qr_image

    table = np.full((h, w, 3), (170, 180, 190), np.uint8)
    cards = {}
    for _, _, code, name in LABELS:
        card = np.full((h, w, 3), (170, 180, 190), np.uint8)
        q = cv2.cvtColor(qr_image(code, 300), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(card, (150, 40), (490, 440), (255, 255, 255), -1)
        card[60:360, 170:470] = cv2.resize(q, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.putText(card, name, (180, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
        cards[code] = card
    with open(path, "wb") as f:
        for i in range(LABEL_SECONDS * FPS):
            t = i / FPS
            img = next((cards[c] for a, b, c, _ in LABELS if a <= t < b), table)
            f.write(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    shop_video(out / "shop.mjpeg")
    label_video(out / "labels.mjpeg")
    print("videos in", out)
