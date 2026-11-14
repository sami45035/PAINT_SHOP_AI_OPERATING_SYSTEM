"""Drawn shop scenes with people the camera can detect (for end-to-end tests and the demo video)."""
import math, cv2, numpy as np

def background(w=960, h=540):
    f = np.full((h, w, 3), (205, 214, 222), np.uint8)
    cv2.rectangle(f, (0, int(h*.62)), (w, h), (150, 160, 170), -1)           # floor
    for i in range(0, w, 160):                                                  # shelves at the back
        cv2.rectangle(f, (i+10, int(h*.12)), (i+150, int(h*.55)), (120, 95, 70), -1)
        for k in range(4):
            y = int(h*.16 + k*h*.095)
            cv2.rectangle(f, (i+14, y), (i+146, y+6), (90, 70, 50), -1)
            for c in range(6):
                col = [(40,60,200),(60,160,220),(200,120,40),(80,170,90),(220,220,220),(60,60,160)][(c+k+i//160)%6]
                cv2.rectangle(f, (i+18+c*21, y-34), (i+34+c*21, y-2), col, -1)
    cv2.rectangle(f, (int(w*.70), int(h*.50)), (int(w*.98), int(h*.68)), (70, 110, 150), -1)   # counter
    return f

def person(f, cx, feet_y, height, shirt, phase=0.0, trousers=(45, 45, 55)):
    hgt = height; u = hgt / 8.0
    head_r = int(u * 0.55)
    hx, hy = int(cx), int(feet_y - hgt + head_r)
    sw = int(u * 1.9)                      # shoulder width (half ~ 0.95u)
    top = hy + head_r + int(u*0.25)
    hip = int(feet_y - hgt * 0.48)
    # legs (walking swing)
    s = math.sin(phase) * u * 0.9
    for sign in (-1, 1):
        x_hip = int(cx + sign * u * 0.4)
        x_foot = int(cx + sign * u * 0.45 + sign * s)
        cv2.line(f, (x_hip, hip), (x_foot, int(feet_y)), trousers, max(3, int(u*0.75)))
        cv2.line(f, (x_foot, int(feet_y)), (x_foot + int(u*0.5), int(feet_y)), (25, 25, 25), max(3, int(u*0.35)))
    # torso
    pts = np.array([[cx - sw//2, top], [cx + sw//2, top], [cx + int(u*0.8), hip + int(u*0.2)], [cx - int(u*0.8), hip + int(u*0.2)]], np.int32)
    cv2.fillPoly(f, [pts], shirt)
    # arms
    for sign in (-1, 1):
        sx = int(cx + sign * sw / 2)
        ex = int(sx + sign * u * 0.15 - sign * s * 0.6)
        cv2.line(f, (sx, top + int(u*0.2)), (ex, hip + int(u*0.1)), shirt, max(3, int(u*0.55)))
        cv2.circle(f, (ex, hip + int(u*0.25)), max(2, int(u*0.28)), (140, 170, 210), -1)
    cv2.line(f, (hx, hy + head_r), (hx, top), (140, 170, 210), max(3, int(u*0.4)))   # neck
    cv2.circle(f, (hx, hy), head_r, (140, 170, 210), -1)                            # face
    cv2.ellipse(f, (hx, hy - int(head_r*0.35)), (head_r, int(head_r*0.7)), 0, 180, 360, (30, 30, 30), -1)  # hair


def scene(t: float, cast, w: int = 960, h: int = 540, bg=None):
    """
    The shop at time t. cast: [(name, shirt_bgr, [(t, x_frac, feet_frac, height_frac), ...])]
    Each person is in the picture between their first and last waypoint, walking between them.
    """
    f = (bg if bg is not None else background(w, h)).copy()
    who = []
    for name, shirt, path in cast:
        if not (path[0][0] <= t <= path[-1][0]):
            continue
        for (t0, x0, y0, s0), (t1, x1, y1, s1) in zip(path, path[1:]):
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                x, y, sz = x0 + a * (x1 - x0), y0 + a * (y1 - y0), s0 + a * (s1 - s0)
                moving = abs(x1 - x0) + abs(y1 - y0) > 1e-3
                person(f, x * w, y * h, sz * h, shirt, phase=(t * 6.0 if moving else 0.0))
                who.append(name)
                break
    return f, who
