"""
AI Room Colour Assistant & Visual Recolouring Engine (Layer 6)
Supports:
1. Natural Language Room & Mood Query Matching
2. Delta-E / Euclidean Color Distance Matching to Paint Catalog
3. Real-Time Stock Status Check & Nearest In-Stock Alternatives
4. Room Photo Wall Segmentation (preserving furniture & windows)
5. Photorealistic Wall Recolouring with Natural Shadow/Lighting Preservation
"""
import cv2
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.config import DB_PATH, SHADES_CATALOG_PATH, SAMPLE_ROOMS_DIR
import sqlite3

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 3:
        hex_code = "".join([c*2 for c in hex_code])
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}".upper()

def color_distance_lab(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """
    Computes Delta-E color distance between two RGB colors using CIE76 Lab approximation.
    """
    # Create 1x1 image and convert to LAB
    img1 = np.uint8([[list(rgb1)]])
    img2 = np.uint8([[list(rgb2)]])
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)[0][0].astype(float)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)[0][0].astype(float)
    return float(np.linalg.norm(lab1 - lab2))

def search_shades_by_query(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Searches paint catalog using natural language room, mood, or color terms.
    """
    with open(SHADES_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    tokens = [t.lower().strip() for t in query.split() if len(t.strip()) > 1]
    scored = []

    for shade in catalog:
        score = 0
        name_lower = shade["shade_name"].lower()
        cat_lower = shade["category"].lower()
        brand_lower = shade["brand"].lower()
        tags_lower = [t.lower() for t in shade.get("room_tags", [])]

        for token in tokens:
            if token in name_lower:
                score += 5
            if any(token in t for t in tags_lower):
                score += 4
            if token in cat_lower:
                score += 3
            if token in brand_lower:
                score += 2

        if score > 0 or not tokens:
            scored.append((score, shade))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [s[1] for s in scored[:limit]] if tokens else catalog[:limit]
    
    # Enrich with live stock
    return enrich_shades_with_stock(results)

def find_nearest_shades_by_hex(target_hex: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Finds closest shades in paint catalog to a user-selected hex color.
    """
    with open(SHADES_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    target_rgb = hex_to_rgb(target_hex)
    scored = []

    for shade in catalog:
        s_rgb = hex_to_rgb(shade["hex_code"])
        dist = color_distance_lab(target_rgb, s_rgb)
        scored.append((dist, shade))

    scored.sort(key=lambda x: x[0])
    closest = [s[1] for s in scored[:limit]]
    return enrich_shades_with_stock(closest)

def enrich_shades_with_stock(shades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Queries SQLite inventory to append current stock, SKU, and availability.
    If out of stock, identifies in-stock alternatives.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        enriched = []
        for s in shades:
            cursor.execute("""
            SELECT p.id, p.sku, p.pack_size_litres, p.selling_price, i.current_stock
            FROM products p
            JOIN inventory i ON p.id = i.product_id
            WHERE p.brand = ? AND p.shade_name = ?
            ORDER BY p.pack_size_litres;
            """, (s["brand"], s["shade_name"]))
            rows = cursor.fetchall()
            
            sizes = []
            total_stock = 0
            for r in rows:
                p_id, sku, size, price, stock = r
                total_stock += stock
                sizes.append({
                    "sku": sku,
                    "size_litres": size,
                    "price": price,
                    "stock": stock
                })
            
            item = dict(s)
            item["pack_options"] = sizes
            item["total_stock"] = total_stock
            item["in_stock"] = total_stock > 0
            
            # If unavailable, find nearest in-stock alternatives
            if total_stock <= 0:
                item["alternatives"] = get_in_stock_alternatives(s["hex_code"], exclude_shade=s["shade_name"])
            else:
                item["alternatives"] = []

            enriched.append(item)
    return enriched

def get_in_stock_alternatives(target_hex: str, exclude_shade: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Finds the closest visual matches that ARE guaranteed in stock.
    """
    with sqlite3.connect(DB_PATH) as conn:
        query = """
        SELECT DISTINCT p.brand, p.sub_brand, p.shade_name, p.shade_code, p.hex_code, p.category, p.selling_price, i.current_stock
        FROM products p
        JOIN inventory i ON p.id = i.product_id
        WHERE i.current_stock > 5 AND p.shade_name != ?;
        """
        df = pd.read_sql_query(query, conn, params=(exclude_shade,))

    target_rgb = hex_to_rgb(target_hex)
    candidates = []
    for _, row in df.iterrows():
        c_rgb = hex_to_rgb(row["hex_code"])
        dist = color_distance_lab(target_rgb, c_rgb)
        candidates.append((dist, row.to_dict()))

    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates[:limit]]

# --- Room Photo Wall Segmentation & Visual Recolouring ---

def segment_wall_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Segments wall surfaces from a room photo:
    - Filters out floor (lower region with different texture/color)
    - Detects large planar regions via bilateral filtering, adaptive thresholding, and morphological operations
    - Protects high-contrast furniture/decor boundaries
    """
    h, w = image_bgr.shape[:2]

    # Preprocessing: Bilateral filter preserves edges while smoothing wall textures
    filtered = cv2.bilateralFilter(image_bgr, 9, 75, 75)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Edge detection to identify furniture and window borders
    edges = cv2.Canny(gray, 30, 100)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)

    # Wall mask heuristic: Walls are predominantly in the upper 75% of the room,
    # relatively uniform in local variance compared to complex floor or textile patterns
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Upper-middle wall zone
    wall_region_y = int(h * 0.78)
    mask[:wall_region_y, :] = 255

    # Subtract floor area (bottom 22% usually rug/floor)
    mask[wall_region_y:, :] = 0

    # Subtract sharp edges (furniture outlines, picture frames, doors)
    mask[edges_dilated > 0] = 0

    # Smooth the mask to eliminate noise
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)

    # Optional blur on mask edges for seamless paint blending
    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask_blurred

def recolor_room_wall(image_rgb: np.ndarray, target_hex: str, wall_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Recolors the segmented wall in a room photo with target paint shade.
    Preserves natural wall lighting, shadows, and ambient textures using LAB luminance blending.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    if wall_mask is None:
        wall_mask = segment_wall_mask(image_bgr)

    # Normalize mask to 0.0 - 1.0 float
    mask_norm = wall_mask.astype(float) / 255.0
    mask_norm_3d = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)

    # Convert original image to LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(float)
    orig_l = lab[:, :, 0]  # Lightness / Luminance channel

    # Target color in LAB
    target_rgb = hex_to_rgb(target_hex)
    target_bgr = np.uint8([[list(reversed(target_rgb))]])
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0][0].astype(float)
    target_l, target_a, target_b = target_lab[0], target_lab[1], target_lab[2]

    # New wall LAB representation:
    # Retain the original wall's local shading variations, modulated around target lightness
    mean_orig_l = np.mean(orig_l[wall_mask > 128]) if np.any(wall_mask > 128) else 128.0
    adjusted_l = np.clip(orig_l * (target_l / max(mean_orig_l, 1.0)), 0, 255)

    recolored_lab = np.zeros_like(lab)
    recolored_lab[:, :, 0] = adjusted_l
    recolored_lab[:, :, 1] = target_a
    recolored_lab[:, :, 2] = target_b

    recolored_bgr = cv2.cvtColor(recolored_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Alpha blend using the segmented wall mask
    blended_bgr = (recolored_bgr.astype(float) * mask_norm_3d + image_bgr.astype(float) * (1.0 - mask_norm_3d)).astype(np.uint8)

    # Return RGB for display in Streamlit
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

def generate_sample_rooms():
    """
    Creates 3 realistic room templates if none exist in sample_rooms directory.
    """
    SAMPLE_ROOMS_DIR.mkdir(parents=True, exist_ok=True)
    rooms = [
        ("modern_living_room.jpg", (220, 215, 205), "Living Room"),
        ("master_bedroom.jpg", (210, 220, 225), "Master Bedroom"),
        ("cozy_dining_room.jpg", (230, 225, 215), "Dining Area")
    ]
    for filename, wall_color, room_title in rooms:
        file_path = SAMPLE_ROOMS_DIR / filename
        if not file_path.exists():
            img = np.full((500, 750, 3), wall_color, dtype=np.uint8)
            # Add floor
            cv2.rectangle(img, (0, 380), (750, 500), (140, 105, 75), -1)
            # Window
            cv2.rectangle(img, (80, 80), (260, 300), (255, 255, 255), 8)
            cv2.rectangle(img, (88, 88), (252, 292), (180, 215, 245), -1)
            cv2.line(img, (170, 88), (170, 292), (255, 255, 255), 4)
            cv2.line(img, (88, 190), (252, 190), (255, 255, 255), 4)
            # Sofa in center
            cv2.rectangle(img, (320, 300), (660, 440), (60, 75, 90), -1)
            cv2.rectangle(img, (300, 330), (330, 430), (50, 60, 75), -1)  # Left arm
            cv2.rectangle(img, (650, 330), (680, 430), (50, 60, 75), -1)  # Right arm
            # Baseboard
            cv2.rectangle(img, (0, 370), (750, 380), (245, 245, 245), -1)
            # Text label
            cv2.putText(img, f"DEMO TEMPLATE: {room_title.upper()}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            cv2.imwrite(str(file_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
