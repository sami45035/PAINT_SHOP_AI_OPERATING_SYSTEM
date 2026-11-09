"""
AI Room Colour Assistant & Visual Recolouring Engine (Layer 6)
Supports:
1. Natural Language Room & Mood Query Matching
2. Delta-E / Euclidean Color Distance Matching to Paint Catalog
3. Real-Time Stock Status Check & Nearest In-Stock Alternatives
Wall finding, recolouring and room-aware suggestions live in src/colour_studio/.
"""
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.config import DB_PATH, SHADES_CATALOG_PATH, SAMPLE_ROOMS_DIR
import re
import sqlite3

SEARCH_STOPWORDS = {
    "for", "the", "and", "with", "colour", "color", "colours", "colors", "shade", "shades", "paint", "paints",
    "want", "need", "suggest", "recommend", "some", "something", "stock", "have", "room", "wall", "walls",
    "please", "show", "give", "nice", "good", "best", "our", "your", "any", "can", "you", "like",
}

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
    import cv2  # loaded only when needed

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

    tokens = [t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 2 and t not in SEARCH_STOPWORDS]
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

# --- Room photo recolouring (kept for older callers; Colour Studio uses src.colour_studio) ---

def recolor_room_wall(image_rgb: np.ndarray, target_hex: str, wall_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Recolour the area in `wall_mask` (bool or 0-255 or 0-1). There is no longer
    a guessed default mask: without a mask the image is returned unchanged.
    """
    from src.colour_studio.recolour import recolour

    if wall_mask is None:
        return image_rgb.copy()
    m = wall_mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    return recolour(image_rgb, m, target_hex)
