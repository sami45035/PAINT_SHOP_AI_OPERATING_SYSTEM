"""
Designer looks: colour prediction the way an interior designer works.

A look is a whole scheme, not one shade — the 60-30-10 rule:
    walls (60%)         the main wall colour
    feature wall (30%)  one wall in a stronger colour (the back wall, if the room has one)
    trim & ceiling (10%) skirting, door frames, ceiling
    accents             two decor colours (cushions, curtains, a vase) to finish it

How a look is chosen for a customer's room:
1. Read the room: what it is (bed -> bedroom, sofa -> living room, ...), its
   undertone (wooden floor and furniture -> warm), how much light it gets, and
   the big furniture colours (a feature wall the colour of the sofa disappears).
2. Score every look against that, plus the style chips and mood words the
   customer picked, and keep a varied top list with one-line reasons.
3. Each paint colour is matched to a shade the shop stocks when one is close
   enough (ΔE < 6); otherwise it's a house-range colour tinted at the counter.

The house range ("Apex House Range", APX- codes) is SAMPLE data invented for
this prototype — swap in the shop's real shade card (data/ or the catalog) for
real use.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.colour_studio.recommend import colour_word, delta_e, hex_to_rgb, is_wood, lab_to_lch, rgb_to_lab

HOUSE_BRAND = "Apex House Range"
TINT_PACKS = [  # sample prices for house-range colours tinted at the counter
    {"size_litres": 1.0, "price": 420.0, "stock": 999, "sku": "APX-T1"},
    {"size_litres": 4.0, "price": 1580.0, "stock": 999, "sku": "APX-T4"},
    {"size_litres": 10.0, "price": 3750.0, "stock": 999, "sku": "APX-T10"},
    {"size_litres": 20.0, "price": 7200.0, "stock": 999, "sku": "APX-T20"},
]
STYLES = ["Modern", "Scandinavian", "Japandi", "Boho", "Indian festive", "Luxe", "Coastal", "Earthy", "Minimal",
          "Playful"]

# (name, mood line, styles, rooms, warmth, moods, [(role, colour name, hex)])
_LOOKS = [
    ("Sage & Oak", "Calm and grounded — soft sage behind the bed, warm linen everywhere else.",
     ["Scandinavian", "Japandi", "Earthy"], ["bedroom", "living"], "warm", ["calm", "fresh", "natural"],
     [("walls", "Soft Linen", "#EDE6DA"), ("feature", "Sage Leaf", "#9CAF88"), ("trim", "Chalk White", "#F7F5F0"),
      ("accent", "Oak", "#B98A5E"), ("accent", "Terracotta Pot", "#C46A4A")]),
    ("Terracotta Sunset", "Sun-baked and cosy — a clay feature wall with sand walls and olive accents.",
     ["Boho", "Earthy"], ["living", "dining"], "warm", ["cozy", "warm", "earthy"],
     [("walls", "Warm Sand", "#E8D5BF"), ("feature", "Terracotta", "#C0643F"), ("trim", "Cream", "#F6EFE4"),
      ("accent", "Olive", "#6B7344"), ("accent", "Mustard", "#D4A43A")]),
    ("Coastal Calm", "Breezy and light — sea-salt walls with a deep harbour-blue wall.",
     ["Coastal", "Scandinavian"], ["bedroom", "bathroom", "living"], "cool", ["calm", "airy", "fresh"],
     [("walls", "Sea Salt", "#E4ECEA"), ("feature", "Harbour Blue", "#5E8299"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Driftwood", "#A89A86"), ("accent", "Sand", "#D8C8A8")]),
    ("Japandi Linen", "Quiet and minimal — rice-paper walls, a mushroom wall, charcoal and walnut details.",
     ["Japandi", "Minimal"], ["bedroom", "living", "study"], "neutral", ["calm", "minimal", "peaceful"],
     [("walls", "Rice Paper", "#F1EBE1"), ("feature", "Mushroom", "#A89C8C"), ("trim", "Bone", "#EAE4D9"),
      ("accent", "Charcoal", "#3E3D3B"), ("accent", "Walnut", "#6E4F3A")]),
    ("Royal Peacock", "Festive and rich — a peacock-teal wall with ivory, gold and magenta.",
     ["Indian festive", "Luxe"], ["living", "dining"], "warm", ["festive", "bold", "elegant"],
     [("walls", "Ivory Silk", "#F3EDE0"), ("feature", "Peacock Teal", "#1F5E63"), ("trim", "Pearl White", "#F8F4EC"),
      ("accent", "Temple Gold", "#C9A24A"), ("accent", "Rani Pink", "#A33A6A")]),
    ("Midnight Study", "Focused and dramatic — a midnight-navy wall, misty grey walls, cognac leather.",
     ["Modern", "Luxe"], ["study", "bedroom", "office"], "cool", ["bold", "elegant", "modern"],
     [("walls", "Mist Grey", "#D9DCDD"), ("feature", "Midnight Navy", "#1F2D44"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Cognac", "#9A5B34"), ("accent", "Brass", "#B8954A")]),
    ("Blush & Brass", "Soft and romantic — petal-pink walls with a dusty-rose wall and brass.",
     ["Modern", "Luxe"], ["bedroom"], "warm", ["romantic", "soft", "elegant"],
     [("walls", "Petal", "#F2E3DF"), ("feature", "Dusty Rose", "#C99A97"), ("trim", "Chalk White", "#F7F5F0"),
      ("accent", "Brass", "#B8954A"), ("accent", "Sage Leaf", "#9CAF88")]),
    ("Olive Grove", "Earthy and relaxed — parchment walls, an olive wall, rust textiles.",
     ["Earthy", "Boho"], ["living", "dining", "study"], "warm", ["earthy", "natural", "cozy"],
     [("walls", "Parchment", "#EFE7D6"), ("feature", "Olive", "#737B4C"), ("trim", "Cream", "#F6EFE4"),
      ("accent", "Rust", "#A4562F"), ("accent", "Linen", "#E5DDCC")]),
    ("Greige Modern", "Clean and current — greige walls, a graphite wall, black and camel.",
     ["Modern", "Minimal"], ["living", "office", "study"], "neutral", ["modern", "clean", "minimal"],
     [("walls", "Greige", "#D8D2C8"), ("feature", "Graphite", "#5B5E61"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Black", "#222222"), ("accent", "Camel", "#B88A5A")]),
    ("Sunny Kitchen", "Cheerful mornings — butter-cream walls, a leaf-green wall, tomato red.",
     ["Playful", "Scandinavian"], ["kitchen", "dining"], "warm", ["cheerful", "bright", "fresh"],
     [("walls", "Butter Cream", "#F4E6C3"), ("feature", "Leaf Green", "#7FA06B"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Tomato", "#D9573B"), ("accent", "Oak", "#B98A5E")]),
    ("Lavender Haze", "Dreamy and soft — lilac-mist walls, a lavender wall, silver and sage.",
     ["Modern", "Playful"], ["bedroom", "kids"], "cool", ["calm", "soft", "dreamy"],
     [("walls", "Lilac Mist", "#E7E1EC"), ("feature", "Lavender", "#A897B8"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Silver Grey", "#A9A9AD"), ("accent", "Sage Leaf", "#9CAF88")]),
    ("Forest Retreat", "Cosy and enveloping — a forest-green wall, foggy walls, leather and brass.",
     ["Luxe", "Earthy"], ["study", "living", "bedroom"], "neutral", ["cozy", "bold", "natural"],
     [("walls", "Fog", "#DCDDD4"), ("feature", "Forest Green", "#2F4A3A"), ("trim", "Off White", "#F2F0E8"),
      ("accent", "Saddle Leather", "#8B5A3C"), ("accent", "Brass", "#B8954A")]),
    ("Sky Playroom", "Bright and happy — cloud walls, a sky-blue wall, sunshine and coral.",
     ["Playful", "Scandinavian"], ["kids"], "cool", ["cheerful", "bright", "playful"],
     [("walls", "Cloud", "#F4F1EA"), ("feature", "Sky Blue", "#8DB8D8"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Sunshine", "#F2C14E"), ("accent", "Coral", "#F07C5C")]),
    ("Desert Clay", "Warm and textured — desert-sand walls, a clay wall, cactus and rattan.",
     ["Boho", "Earthy"], ["living", "bedroom"], "warm", ["warm", "earthy", "relaxed"],
     [("walls", "Desert Sand", "#E2CFB5"), ("feature", "Clay", "#B77B5B"), ("trim", "Cream", "#F6EFE4"),
      ("accent", "Cactus", "#6E8B6A"), ("accent", "Rattan", "#C7A46F")]),
    ("Charcoal Gallery", "Crisp and graphic — white walls with one charcoal wall; let art and wood pop.",
     ["Modern", "Minimal"], ["living", "office", "study"], "neutral", ["modern", "bold", "clean"],
     [("walls", "Gallery White", "#F7F7F5"), ("feature", "Charcoal", "#3B3E42"), ("trim", "Gallery White", "#F7F7F5"),
      ("accent", "Oak", "#B98A5E"), ("accent", "Black", "#222222")]),
    ("Ivory & Gold", "Classic and auspicious — ivory walls, a champagne wall, gold and maroon.",
     ["Indian festive", "Luxe"], ["living", "puja", "dining"], "warm", ["festive", "elegant", "traditional"],
     [("walls", "Ivory", "#F5EEDD"), ("feature", "Champagne", "#E6D3AE"), ("trim", "Pearl White", "#F8F4EC"),
      ("accent", "Temple Gold", "#C9A24A"), ("accent", "Maroon", "#7A2A2E")]),
    ("Mint Fresh", "Clean and fresh — mint-cream walls, a mint wall, coral and wood.",
     ["Scandinavian", "Playful"], ["kitchen", "bathroom", "kids"], "cool", ["fresh", "bright", "clean"],
     [("walls", "Mint Cream", "#E6F1EA"), ("feature", "Mint", "#A8D5BA"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Coral", "#F07C5C"), ("accent", "Oak", "#B98A5E")]),
    ("Aubergine Night", "Moody and luxurious — a deep aubergine wall, smoky walls, velvet pink and gold.",
     ["Luxe", "Modern"], ["bedroom", "living"], "cool", ["bold", "dramatic", "elegant"],
     [("walls", "Smoke", "#D7D2D3"), ("feature", "Aubergine", "#4E2E3F"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Velvet Pink", "#C98A9B"), ("accent", "Temple Gold", "#C9A24A")]),
    ("Indigo & Ochre", "Modern Indian — an indigo wall on rice-white, ochre and block-print red.",
     ["Indian festive", "Boho"], ["living", "bedroom"], "neutral", ["bold", "festive", "artistic"],
     [("walls", "Rice White", "#F2EDE3"), ("feature", "Indigo", "#2E3F6E"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Ochre", "#C8902F"), ("accent", "Block-print Red", "#A33B2B")]),
    ("Stone & Rust", "Warm industrial — stone walls, a rust wall, cream and black.",
     ["Modern", "Earthy"], ["living", "dining", "office"], "warm", ["bold", "warm", "modern"],
     [("walls", "Stone", "#CFC8BC"), ("feature", "Rust", "#9C4A2C"), ("trim", "Off White", "#F2F0E8"),
      ("accent", "Cream", "#F6EFE4"), ("accent", "Black", "#222222")]),
    ("Powder Blue Rest", "Restful and light — powder-blue walls with a denim wall.",
     ["Coastal", "Scandinavian"], ["bedroom", "kids"], "cool", ["calm", "airy", "peaceful"],
     [("walls", "Powder Blue", "#D6E4EE"), ("feature", "Denim", "#4F6D8C"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Sand", "#D8C8A8"), ("accent", "Chalk White", "#F7F5F0")]),
    ("Warm White Classic", "Safe, bright and timeless — warm-white walls with an oatmeal wall. Suits any room.",
     ["Minimal", "Scandinavian"], ["living", "bedroom", "dining", "study", "kitchen", "office"], "neutral",
     ["bright", "clean", "airy", "light"],
     [("walls", "Warm White", "#F4EFE6"), ("feature", "Oatmeal", "#DCCFBD"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Oak", "#B98A5E"), ("accent", "Sage Leaf", "#9CAF88")]),
    ("Emerald Luxe", "Jewel-toned and glamorous — an emerald wall, pearl walls, gold and blush.",
     ["Luxe", "Indian festive"], ["living", "dining", "bedroom"], "neutral", ["bold", "elegant", "festive"],
     [("walls", "Pearl", "#EEECE6"), ("feature", "Emerald", "#1B5E4A"), ("trim", "Pure White", "#FAFAF8"),
      ("accent", "Temple Gold", "#C9A24A"), ("accent", "Blush", "#E8C5C8")]),
    ("Peach Glow", "Sunny and soft — peach-cream walls, an apricot wall, teal accents.",
     ["Playful", "Boho"], ["bedroom", "kids", "living"], "warm", ["cheerful", "warm", "soft"],
     [("walls", "Peach Cream", "#F6DCC8"), ("feature", "Apricot", "#E8A07A"), ("trim", "Cream", "#F6EFE4"),
      ("accent", "Teal", "#2F7F7A"), ("accent", "Cream", "#F6EFE4")]),
]

ROOM_CLUES = {  # ADE20K label -> room type
    "bed": "bedroom", "pillow": "bedroom", "wardrobe": "bedroom", "sofa": "living", "armchair": "living",
    "coffee table": "living", "television receiver": "living", "fireplace": "living", "table": "dining",
    "chair": "dining", "stove": "kitchen", "refrigerator": "kitchen", "kitchen island": "kitchen",
    "dishwasher": "kitchen", "sink": "kitchen", "bathtub": "bathroom", "toilet": "bathroom", "shower": "bathroom",
    "desk": "study", "bookcase": "study", "computer": "office",
}
ROOM_WORDS = {"bedroom": "bedroom", "bed room": "bedroom", "living": "living", "hall": "living", "drawing": "living",
              "dining": "dining", "kitchen": "kitchen", "bath": "bathroom", "kids": "kids", "children": "kids",
              "nursery": "kids", "study": "study", "office": "office", "puja": "puja", "pooja": "puja"}


def _code(name: str) -> str:
    return "APX-" + str(int(hashlib.md5(name.encode()).hexdigest()[:6], 16) % 9000 + 1000)


def _lrv(hex_code: str) -> float:
    """Light reflectance value (0 = black, 100 = white): how much light a paint gives back."""
    r, g, b = [c / 255 for c in hex_to_rgb(hex_code)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return round((0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]) * 100, 1)


def looks() -> List[Dict[str, Any]]:
    out = []
    for name, mood, styles, rooms, warmth, moods, colours in _LOOKS:
        paints = {role: {"name": n, "hex": h} for role, n, h in colours if role != "accent"}
        accents = [{"name": n, "hex": h} for role, n, h in colours if role == "accent"]
        out.append({"id": name.lower().replace(" & ", "-").replace(" ", "-"), "name": name, "mood": mood,
                    "styles": styles, "rooms": rooms, "warmth": warmth, "moods": moods, "paints": paints,
                    "accents": accents})
    return out


# ----------------------------------------------------------------------------- shades we can sell

def house_shade(name: str, hex_code: str) -> Dict[str, Any]:
    return {"brand": HOUSE_BRAND, "sub_brand": "Tinted at the counter", "shade_name": name, "shade_code": _code(name),
            "hex_code": hex_code, "category": "Interior Emulsion", "finish": "Smooth Matt", "room_tags": [],
            "total_stock": 999, "in_stock": True, "tinted": True, "pack_options": [dict(p) for p in TINT_PACKS]}


def match_paint(name: str, hex_code: str, stocked: List[Dict[str, Any]], max_de: float = 6.0) -> Dict[str, Any]:
    """The shop's own stocked shade if one is close enough, else the house-range colour tinted to order."""
    lab = rgb_to_lab(hex_to_rgb(hex_code))[0]
    best, best_d = None, max_de
    for s in stocked:
        d = delta_e(lab, rgb_to_lab(hex_to_rgb(s["hex_code"]))[0])
        if d < best_d and int(s.get("total_stock", 0)) > 0:
            best, best_d = s, d
    if best is not None:
        return {**best, "matched_from": name, "match_de": round(best_d, 1)}
    return house_shade(name, hex_code)


# ----------------------------------------------------------------------------- reading the room

def detect_room(label_map: Optional[np.ndarray], id2label: Dict[int, str]) -> Optional[str]:
    if label_map is None or not id2label:
        return None
    total = label_map.size
    votes: Dict[str, float] = {}
    counts = np.bincount(label_map.ravel(), minlength=max(id2label) + 1)
    for cid, name in id2label.items():
        room = ROOM_CLUES.get(name.strip())
        if room and cid < len(counts) and counts[cid] / total > 0.01:
            votes[room] = votes.get(room, 0) + counts[cid] / total
    return max(votes, key=votes.get) if votes else None


def room_context(image_rgb: np.ndarray, palette: List[Dict[str, Any]], wall_now: Optional[Dict[str, Any]],
                 label_map=None, id2label=None, n_walls: int = 1, text: str = "") -> Dict[str, Any]:
    """Everything the scoring needs to know about this room."""
    words = (text or "").lower()
    room = next((r for w, r in ROOM_WORDS.items() if w in words), None) or detect_room(label_map, id2label or {})
    wood = any(is_wood(p["lch"]) or p["name"].startswith("wooden") for p in palette)
    warm_items = sum(1 for p in palette if p["lch"][1] >= 10 and (p["lch"][2] < 100 or p["lch"][2] > 330))
    cool_items = sum(1 for p in palette if p["lch"][1] >= 10 and 150 <= p["lch"][2] <= 300)
    undertone = "warm" if (wood or warm_items > cool_items) else ("cool" if cool_items > warm_items else "neutral")
    # light: how bright the walls look in the photo (median lightness of the wall)
    light = "normal"
    if wall_now is not None:
        L = float(wall_now["lab"][0])
        light = "dim" if L < 55 else ("bright" if L > 80 else "normal")
    big = [p for p in palette if p.get("is_big_furniture")]
    return {"room": room, "undertone": undertone, "wood": wood, "light": light, "big_furniture": big,
            "n_walls": n_walls, "palette": palette}


# ----------------------------------------------------------------------------- scoring

ROOM_NAMES = {"living": "living room", "dining": "dining room", "kids": "kids' room", "puja": "puja room",
              "office": "home office", "study": "study", "bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom"}


def _hue_gap(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def _harmony(look: Dict[str, Any], palette: List[Dict[str, Any]]) -> Tuple[float, str]:
    """How the look's feature wall and accents work with the coloured things already in the room."""
    best, text = 0.0, ""
    colours = [("feature wall", look["paints"]["feature"]["hex"])] + [(a["name"].lower(), a["hex"]) for a in look["accents"]]
    for item in palette:
        L, C, H = item["lch"]
        if C < 15 or is_wood(item["lch"]):
            continue                       # whites, greys and wood go with everything
        for role, hx in colours:
            l2, c2, h2 = lab_to_lch(rgb_to_lab(hex_to_rgb(hx)))[0]
            if c2 < 12:
                continue
            gap = _hue_gap(H, h2)
            if gap < 30:
                sc, t = 0.6, f"the {role} echoes the {item['name']}"
            elif 140 <= gap <= 220:
                sc, t = 0.7, f"the {role} plays off the {item['name']}"
            else:
                continue
            if sc > best:
                best, text = sc, t
    return best, text


def score_look(look: Dict[str, Any], ctx: Dict[str, Any], styles: List[str], mood_words: List[str]) -> Dict[str, Any]:
    s, why, warn = 0.0, [], []
    if ctx["room"] and ctx["room"] in look["rooms"]:
        s += 1.2
        why.append(f"made for a {ROOM_NAMES.get(ctx['room'], ctx['room'])}")
    h, htxt = _harmony(look, ctx.get("palette", []))
    if h:
        s += h
        why.insert(0, htxt)
    if look["warmth"] == ctx["undertone"]:
        s += 1.0
        why.append("warm tones to match the wood" if ctx["wood"] and look["warmth"] == "warm"
                   else f"{look['warmth']} tones like the rest of the room")
    elif look["warmth"] == "neutral":
        s += 0.6
        why.append("neutral, so it sits easily with what's already there")
    elif ctx["undertone"] == "warm" and look["warmth"] == "cool":
        s -= 0.2
    walls_lrv = _lrv(look["paints"]["walls"]["hex"])
    feature_lrv = _lrv(look["paints"]["feature"]["hex"])
    if ctx["light"] == "dim":
        if walls_lrv >= 70:
            s += 0.6
            why.append("light walls bounce light around a dim room")
        else:
            s -= 0.6
            warn.append("walls this deep will make a dim room darker")
    elif ctx["light"] == "bright" and feature_lrv < 20:
        s += 0.4
        why.append("plenty of light to carry a deep feature wall")
    if ctx["n_walls"] < 2 and feature_lrv < 25:
        s -= 0.3       # one wall visible: no feature wall, the deep colour is wasted
    feat_lab = rgb_to_lab(hex_to_rgb(look["paints"]["feature"]["hex"]))[0]
    for item in ctx["big_furniture"]:
        if delta_e(feat_lab, item["lab"]) < 15:
            s -= 1.0
            warn.append(f"the feature wall is close to the {item['name']}, so it may disappear behind it")
    hit = [st for st in styles if st in look["styles"]]
    if hit:
        s += 2.0 * len(hit)
        why.insert(0, " & ".join(hit) + " style")
    moods = [m for m in mood_words if m in look["moods"]]
    if moods:
        s += 0.8 * len(moods)
        why.insert(0, f"{', '.join(moods)} feel")
    if ctx["wood"] and any(a["name"] in ("Oak", "Walnut", "Rattan", "Saddle Leather", "Cognac") for a in look["accents"]):
        s += 0.2
    reason = "; ".join(why[:2]) or "a well-balanced scheme"
    reason = reason[0].upper() + reason[1:]
    if warn:
        reason += f". Note: {warn[0]}"
    return {**look, "score": round(s, 2), "reason": reason, "walls_lrv": walls_lrv, "feature_lrv": feature_lrv}


def mood_words_from(text: str) -> List[str]:
    vocab = {m for lk in _LOOKS for m in lk[5]}
    syn = {"cosy": "cozy", "relaxing": "calm", "soothing": "calm", "peace": "peaceful", "luxury": "elegant",
           "royal": "elegant", "happy": "cheerful", "fun": "playful", "dark": "bold", "moody": "dramatic",
           "nature": "natural", "simple": "minimal", "festival": "festive", "diwali": "festive"}
    out = []
    for w in (text or "").lower().replace(",", " ").split():
        w = syn.get(w, w)
        if w in vocab and w not in out:
            out.append(w)
    return out


def recommend_looks(ctx: Dict[str, Any], styles: List[str], text: str = "", limit: int = 6) -> List[Dict[str, Any]]:
    """Top looks for this room, varied (no two with near-identical feature walls)."""
    words = mood_words_from(text)
    scored = sorted((score_look(lk, ctx, styles, words) for lk in looks()), key=lambda x: -x["score"])
    picks: List[Dict[str, Any]] = []
    for lk in scored:
        lab = rgb_to_lab(hex_to_rgb(lk["paints"]["feature"]["hex"]))[0]
        if all(delta_e(lab, rgb_to_lab(hex_to_rgb(p["paints"]["feature"]["hex"]))[0]) > 12 for p in picks):
            picks.append(lk)
        if len(picks) == limit:
            break
    return picks


def wall_plan(regions, look: Dict[str, Any]) -> Dict[int, str]:
    """Which wall gets which paint: the back wall (or the biggest) is the feature wall when there are 2+ walls."""
    if not regions:
        return {}
    main, feat = look["paints"]["walls"]["hex"], look["paints"]["feature"]["hex"]
    if len(regions) < 2:
        return {regions[0].id: main}
    back = [r for r in regions if r.name.startswith("Back")]
    feature_wall = back[0] if back else max(regions, key=lambda r: r.area_share)
    return {r.id: (feat if r.id == feature_wall.id else main) for r in regions}


def describe(look: Dict[str, Any]) -> str:
    p = look["paints"]
    return (f"{p['walls']['name']} walls, a {p['feature']['name']} feature wall, {p['trim']['name']} trim; "
            f"accents in {look['accents'][0]['name'].lower()} and {look['accents'][1]['name'].lower()}")


def palette_word(hex_code: str) -> str:
    return colour_word(lab_to_lch(rgb_to_lab(hex_to_rgb(hex_code)))[0])
