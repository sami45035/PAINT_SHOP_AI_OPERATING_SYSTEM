"""Custom Streamlit components shipped with the app (plain HTML/JS, no build step)."""
import base64
import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

_DIR = Path(__file__).parent
_wall_brush = None


def _component():
    global _wall_brush
    if _wall_brush is None:
        import streamlit.components.v1 as components

        _wall_brush = components.declare_component("wall_brush", path=str(_DIR / "wall_brush"))
    return _wall_brush


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
