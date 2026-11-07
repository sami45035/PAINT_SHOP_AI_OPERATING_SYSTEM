"""
Photo-real wall recolouring.

A wall's appearance is roughly paint colour × light falling on it. We estimate
the light from the photo (the wall's brightness relative to its typical
brightness), then multiply the new paint colour by that light — in linear RGB,
which is how light actually mixes. Shadows, corners, texture and the glow near
a window all carry over, and the well-lit part of the wall shows the paint's
true catalogue colour.
"""
from typing import Tuple

import cv2
import numpy as np


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    h = hex_code.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _luminance(lin_rgb: np.ndarray) -> np.ndarray:
    return lin_rgb[..., 0] * 0.2126 + lin_rgb[..., 1] * 0.7152 + lin_rgb[..., 2] * 0.0722


def recolour(image_rgb: np.ndarray, alpha: np.ndarray, paint_hex: str, keep_texture: float = 1.0) -> np.ndarray:
    """
    Paint the area given by `alpha` (float 0..1, same height/width as the image).

    keep_texture: 1.0 keeps all of the photo's light and texture; lower values
    flatten it toward a solid colour (useful for very patterned old walls).
    """
    if alpha.shape[:2] != image_rgb.shape[:2]:
        alpha = cv2.resize(alpha.astype(np.float32), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    if a.max() <= 0:
        return image_rgb.copy()

    lin = srgb_to_linear(image_rgb.astype(np.float32) / 255.0)
    Y = _luminance(lin)

    # Typical wall brightness: median of clearly-selected pixels (robust to lamps and dark corners)
    core = a > 0.6
    ref = float(np.median(Y[core])) if core.any() else float(np.average(Y, weights=a + 1e-6))
    ref = max(ref, 1e-4)

    light = (Y / ref).astype(np.float32)
    # Very fine noise (JPEG blocks) adds grain to dark paints — smooth the light map a touch
    light = cv2.bilateralFilter(light, 5, 0.08, 3)
    light = 1.0 + (light - 1.0) * keep_texture
    light = np.clip(light, 0.0, 2.5)

    paint_lin = srgb_to_linear(np.array(hex_to_rgb(paint_hex), dtype=np.float32) / 255.0)
    painted_lin = light[..., None] * paint_lin[None, None, :]

    # Bright spots: instead of clipping each channel (which shifts the hue), fade
    # toward white the way a real over-lit wall does.
    peak = painted_lin.max(axis=2, keepdims=True)
    over = peak > 1.0
    if over.any():
        safe_peak = np.maximum(peak, 1.0)
        t = 1.0 - 1.0 / safe_peak
        rolled = (painted_lin / safe_peak) * (1.0 - t) + t
        painted_lin = np.where(over, rolled, painted_lin)
    painted_lin = np.clip(painted_lin, 0.0, 1.0)

    painted = linear_to_srgb(painted_lin) * 255.0
    out = painted * a[..., None] + image_rgb.astype(np.float32) * (1.0 - a[..., None])
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def side_by_side(before: np.ndarray, after: np.ndarray, gap: int = 8) -> np.ndarray:
    """Before | after image for saving or sharing on WhatsApp."""
    h = before.shape[0]
    spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.concatenate([before, spacer, after], axis=1)


def outline_overlay(image_rgb: np.ndarray, masks_and_colours, thickness: int = 2) -> np.ndarray:
    """Draw wall outlines (for the 'pick a wall' view). masks_and_colours: [(bool mask, (r,g,b), fill_alpha)]."""
    out = image_rgb.copy()
    for mask, colour, fill in masks_and_colours:
        m8 = mask.astype(np.uint8)
        if fill > 0:
            tint = np.zeros_like(out)
            tint[:] = colour
            out = np.where(m8[..., None] > 0, (out * (1 - fill) + tint * fill).astype(np.uint8), out)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, thickness, lineType=cv2.LINE_AA)
    return out
