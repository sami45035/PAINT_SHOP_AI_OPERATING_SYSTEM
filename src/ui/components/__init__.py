"""Custom Streamlit components shipped with the app (plain HTML/JS, no build step)."""
import base64
import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

_DIR = Path(__file__).parent
_declared = {}


def _component(name: str = "wall_brush"):
    if name not in _declared:
        import streamlit.components.v1 as components

        _declared[name] = components.declare_component(name, path=str(_DIR / name))
    return _declared[name]


def to_data_url(arr: np.ndarray, fmt: str = "JPEG", quality: int = 90) -> str:
    """numpy image (RGB or single channel) -> data: URL."""
    buf = io.BytesIO()
    im = Image.fromarray(arr)
    if fmt == "JPEG":
        im.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=False, compress_level=3)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def decode_edits(data_url: str, shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Brush layer PNG (red = painted, green = erased) -> two float masks 0..1 at `shape` (h, w)."""
    if not data_url or "," not in data_url:
        return None
    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    if arr.shape[:2] != tuple(shape):
        arr = np.array(Image.fromarray(arr).resize((shape[1], shape[0]), Image.BILINEAR))
    return arr[..., 0].astype(np.float32) / 255.0, arr[..., 1].astype(np.float32) / 255.0


def apply_edits(alpha: np.ndarray, edits: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Same rule as the browser: erased paint comes off, brushed paint goes on."""
    if edits is None:
        return alpha
    add, erase = edits
    return np.maximum(alpha * (1.0 - erase), add).astype(np.float32)


def wall_brush(*, key: str, photo_key: str, orig_url: str, painted_url: str, base_url: str, width: int, height: int,
               edits_url: str = "", reset: int = 0, size: int = 36):
    """Paint / erase touch-ups drawn in the browser. Returns {'key': photo key, 'edits': PNG data URL, 'strokes', 'ts'} or None."""
    return _component()(key=key, default=None, **{
        "photo": photo_key, "orig": orig_url, "painted": painted_url, "base": base_url,
        "width": int(width), "height": int(height), "edits": edits_url, "reset": int(reset), "size": int(size),
    })


def browser_cam(*, key: str, ack: int, boxes, line: float, zone, interval_ms: int = 450, width: int = 480,
                mode: str = "door", door_x=(0.0, 1.0)):
    """
    Live camera in the visitor's browser (works on Streamlit Cloud). Sends a small
    JPEG about twice a second: returns {'seq', 'frame' (data URL), 'ts'} or None.
    `ack` = last seq processed, so the browser never queues frames faster than we can count.
    """
    return _component("browser_cam")(key=key, default=None, ack=int(ack), boxes=boxes or [], line=float(line),
                                     zone=[float(v) for v in zone], interval=int(interval_ms), width=int(width),
                                     mode=mode, door_x1=float(door_x[0]), door_x2=float(door_x[1]))


def decode_image(data_url: str):
    """data: URL (JPEG/PNG) -> BGR numpy image for OpenCV, or None."""
    if not data_url or "," not in data_url:
        return None
    import cv2

    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def code_scanner(*, key: str, cart: Optional[dict] = None, done: Optional[dict] = None, notice: Optional[dict] = None,
                 payment: str = "Cash", payments: Optional[list] = None):
    """
    The counter's scanner: camera, the bill being made, and the saved bill.
    cart    -> {'count', 'total', 'lines': [{'pid', 'qty', 'label', 'hex', 'price', 'amount', 'in_stock', 'short'}]}
    done    -> the bill just saved {'invoice', 'time', 'total', 'payment', 'lines', 'wa_text'}: camera off, receipt shown
    notice  -> a short message over the camera {'id', 'text', 'kind': 'ok'|'warn'}
    Returns the latest event: a scan {'code', 'format', 'ts'} or {'action': save|undo|inc|dec|next|phone, ..., 'ts'}.
    """
    return _component("code_scanner")(key=key, default=None, cart=cart, done=done, notice=notice,
                                      payment=payment, payments=payments)
