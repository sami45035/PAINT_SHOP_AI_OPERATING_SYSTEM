"""
Cut the filmed clips (tests/browser/record_demo.py) into the presentation video:
title card, each scene with a caption, the phone shown in a phone frame, end card.

Run:  python tools/edit_demo.py CLIPS_DIR OUT.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
TEAL, TEAL_DARK, INK, SLATE, SAND = (11, 110, 121), (8, 70, 78), (21, 32, 43), (90, 100, 114), (238, 228, 212)
FONT_DIR = Path("/usr/share/fonts/opentype/inter")
CAPTION_S = 7.5          # a caption stays this long, then fades so the screen is fully visible
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]

# scene: (clip, kind, title, line)   kind: laptop | phone
SCENES = [
    ("overview", "laptop", "Overview", "Today's sales, stock, visitors and alerts — at a glance"),
    ("forecast", "laptop", "AI Sales Predictor",
     "Learns weekday and festival effects from the shop's own sales — tested on the past: about half the error of a simple average"),
    ("colour", "laptop", "Colour Prediction · Colour Studio",
     "Any stocked shade on the customer's own wall — and designer palettes matched to shades in stock"),
    ("__counter", "card", "At the counter", "Who walked in · who bought · why the others didn't"),
    ("camera_setup", "laptop", "Staff uniform", "Staff in the shop's uniform are recognised and never counted as customers"),
    ("camera_live", "laptop", "Store Camera — live",
     "Customers counted as they come in; staff tagged \"not counted\"  ·  camera feed simulated with drawn people, same code as a real webcam"),
    ("scan", "phone", "Scan & Bill",
     "Scan each can's QR label|Pay by Cash, UPI, Card or Credit|Save: stock goes down by one per can|Send the bill on WhatsApp"),
    ("camera_b", "laptop", "Next customer", "A second customer comes to the counter — and leaves without buying"),
    ("asking", "phone", "Didn't buy? One tap", "The reason is matched to the customer at the counter|Shades asked for but not stocked become \"Lost sales\" alerts"),
    ("buyers", "laptop", "Who bought, who didn't — and why",
     "Every bill matched to a visitor: 2 came in, 1 bought, 1 \"just asking\" — the staff member wasn't counted"),
    ("stock", "laptop", "Stock & pre-order", "What to order, how much and by when — sized by the forecast"),
    ("alerts", "laptop", "Smart alerts", "Running out · lost sales · shelves to refill · theft · expiring stock · money stuck"),
    ("aura", "laptop", "Ask Aura", "Ask anything about the shop in plain words"),
]


def font(size, weight="Bold"):
    for name in (f"Inter-{weight}.otf", f"InterDisplay-{weight}.otf", "Inter-Regular.otf"):
        p = FONT_DIR / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)


def wrap(draw, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    return lines + ([cur] if cur else [])


def run(cmd):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cmd], check=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def logo(draw, x, y, s=1.0):
    draw.rounded_rectangle([x, y, x + 64 * s, y + 64 * s], radius=int(14 * s), fill=TEAL)
    for i, (hh, col) in enumerate(((30, (255, 255, 255)), (44, (232, 196, 150)), (22, (255, 255, 255)))):
        bx = x + (14 + i * 14) * s
        draw.rounded_rectangle([bx, y + (52 - hh) * s, bx + 9 * s, y + 52 * s], radius=int(4 * s), fill=col)


# ----------------------------------------------------------------------------- pictures
def caption_png(path, title, line):
    """A lower-third card over the laptop scenes."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ft, fl = font(40, "Bold"), font(27, "Medium")
    lines = wrap(d, line, fl, 1180)
    h = 46 + 50 + 38 * len(lines) + 28
    x0, y0 = 56, H - h - 52
    w = max(d.textlength(title, font=ft), *(d.textlength(t, font=fl) for t in lines)) + 112
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([x0 + 4, y0 + 10, x0 + w + 4, y0 + h + 10], radius=26, fill=(0, 0, 0, 70))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=26, fill=(255, 255, 255, 246))
    d.rounded_rectangle([x0, y0, x0 + 12, y0 + h], radius=6, fill=TEAL + (255,))
    d.text((x0 + 48, y0 + 34), title, font=ft, fill=INK)
    for i, t in enumerate(lines):
        d.text((x0 + 48, y0 + 34 + 58 + i * 38), t, font=fl, fill=SLATE)
    img.save(path)


def gradient():
    g = Image.new("RGB", (W, H), TEAL_DARK)
    px = g.load()
    for y in range(H):
        for x in range(0, W, 4):
            t = 0.55 * x / W + 0.45 * y / H
            c = tuple(int(TEAL[i] * (1 - t) + TEAL_DARK[i] * t) for i in range(3))
            for k in range(4):
                if x + k < W:
                    px[x + k, y] = c
    return g


def phone_bg(path, mask_path, title, line, size):
    """Branded background with an empty phone frame (the video goes inside) and the caption beside it."""
    pw, ph = size
    img = gradient().convert("RGBA")
    d = ImageDraw.Draw(img)
    px, py = 300, (H - ph) // 2
    bez = 18
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([px - bez + 10, py - bez + 24, px + pw + bez + 10, py + ph + bez + 24],
                                            radius=64, fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(26)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([px - bez, py - bez, px + pw + bez, py + ph + bez], radius=60, fill=(17, 20, 24))
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=44, fill=(244, 245, 247))
    tx = px + pw + 170
    logo(d, tx, 250, 1.0)
    d.text((tx + 86, 262), "Apex Colors & Paints", font=font(30, "SemiBold"), fill=(255, 255, 255))
    d.text((tx, 360), title, font=font(72, "Bold"), fill=(255, 255, 255))
    y = 480
    for k, part in enumerate(line.split("|")):
        lines = wrap(d, part, font(34, "Medium"), W - tx - 110)
        d.ellipse([tx, y + 12, tx + 16, y + 28], fill=SAND)
        for t in lines:
            d.text((tx + 40, y), t, font=font(34, "Medium"), fill=(234, 244, 245))
            y += 48
        y += 22
    img.convert("RGB").save(path)
    m = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, pw - 1, ph - 1], radius=44, fill=255)
    m.save(mask_path)
    return px, py


def title_png(path):
    img = gradient()
    d = ImageDraw.Draw(img)
    logo(d, 160, 250, 1.6)
    d.text((290, 262), "Apex Colors & Paints", font=font(40, "SemiBold"), fill=(255, 255, 255))
    d.text((160, 400), "Paint Shop AI", font=font(120, "Bold"), fill=(255, 255, 255))
    d.text((160, 540), "Operating System", font=font(120, "Bold"), fill=SAND)
    d.text((164, 720), "AI sales predictor  ·  colour prediction  ·  Colour Studio  ·  Scan & Bill  ·  Store Camera",
           font=font(36, "Medium"), fill=(220, 236, 238))
    d.text((164, 790), "Value proposition design for paint shop owners — working prototype",
           font=font(32, "Regular"), fill=(190, 216, 219))
    img.save(path)


def card_png(path, title, line):
    img = gradient()
    d = ImageDraw.Draw(img)
    tw = d.textlength(title, font=font(96, "Bold"))
    d.text(((W - tw) / 2, 420), title, font=font(96, "Bold"), fill=(255, 255, 255))
    lw = d.textlength(line, font=font(40, "Medium"))
    d.text(((W - lw) / 2, 560), line, font=font(40, "Medium"), fill=SAND)
    img.save(path)


def end_png(path):
    img = gradient()
    d = ImageDraw.Draw(img)
    d.text((160, 200), "What the shop owner gets", font=font(76, "Bold"), fill=(255, 255, 255))
    rows = [("Order the right stock before festivals", "AI forecast learned from the shop's own sales"),
            ("Show customers their wall in any shade", "Colour Studio with designer palettes in stock"),
            ("Know who walks in, who buys — and why not", "Camera + Scan & Bill, staff never counted"),
            ("Fewer stock-outs · more sales · less loss", "Alerts for running out, lost sales and theft")]
    y = 360
    for big, small in rows:
        d.ellipse([162, y + 16, 186, y + 40], fill=SAND)
        d.text((220, y), big, font=font(46, "SemiBold"), fill=(255, 255, 255))
        d.text((220, y + 62), small, font=font(31, "Regular"), fill=(200, 224, 227))
        y += 150
    d.text((160, 980), "Thank you", font=font(40, "Bold"), fill=SAND)
    img.save(path)


# ----------------------------------------------------------------------------- scenes
def still(png, out, secs):
    tmp = out.with_name(out.stem + ".part.mp4")
    run(["-framerate", str(FPS), "-i", str(png), "-vf",
         f"format=yuv420p,loop=-1:1,trim=duration={secs},fade=t=in:st=0:d=0.5,fade=t=out:st={secs - 0.5}:d=0.5",
         *ENC, str(tmp)])
    tmp.rename(out)


def laptop_scene(clip, cap, out):
    d = duration(clip)
    tmp = out.with_name(out.stem + ".part.mp4")
    run(["-i", str(clip), "-framerate", str(FPS), "-i", str(cap), "-filter_complex",
         f"[1:v]format=rgba,loop=-1:1,trim=duration={d},fade=t=in:st=0.4:d=0.5:alpha=1"
         f"{f',fade=t=out:st={CAPTION_S}:d=0.6:alpha=1' if d > CAPTION_S + 2.5 else ''}[c];"
         f"[0:v]scale={W}:{H}[v];[v][c]overlay=0:0:format=auto,"
         f"fade=t=in:st=0:d=0.35,fade=t=out:st={max(0, d - 0.35)}:d=0.35,format=yuv420p",
         "-t", str(d), *ENC, str(tmp)])
    tmp.rename(out)


def phone_scene(clip, bg, mask, pos, size, out):
    d = duration(clip)
    pw, ph = size
    tmp = out.with_name(out.stem + ".part.mp4")
    run(["-framerate", str(FPS), "-i", str(bg), "-i", str(clip), "-framerate", str(FPS), "-i", str(mask), "-filter_complex",
         f"[0:v]loop=-1:1,trim=duration={d}[b];[2:v]loop=-1:1,trim=duration={d},format=gray[m];"
         f"[1:v]scale={pw}:{ph}:flags=lanczos,format=rgba[p];[p][m]alphamerge[pm];"
         f"[b][pm]overlay={pos[0]}:{pos[1]}:format=auto,"
         f"fade=t=in:st=0:d=0.35,fade=t=out:st={max(0, d - 0.35)}:d=0.35,format=yuv420p",
         "-t", str(d), *ENC, str(tmp)])
    tmp.rename(out)


def main():
    clips = Path(sys.argv[1])
    out = Path(sys.argv[2])
    work = out.parent / "edit_work"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    title_png(work / "title.png")
    still(work / "title.png", work / "00_title.mp4", 6.0)
    parts.append(work / "00_title.mp4")
    for i, (name, kind, title, line) in enumerate(SCENES, 1):
        part = work / f"{i:02d}_{name.strip('_')}.mp4"
        if kind == "card":
            card_png(work / f"{name}.png", title, line)
            still(work / f"{name}.png", part, 3.5)
        else:
            clip = clips / f"{name}.mp4"
            if not clip.exists():
                print("missing clip:", name)
                continue
            if kind == "laptop":
                caption_png(work / f"cap_{name}.png", title, line)
                laptop_scene(clip, work / f"cap_{name}.png", part)
            else:
                size = (int(540 * 0.82), int(1170 * 0.82) // 2 * 2)
                pos = phone_bg(work / f"bg_{name}.png", work / f"mask_{name}.png", title, line, size)
                phone_scene(clip, work / f"bg_{name}.png", work / f"mask_{name}.png", pos, size, part)
        parts.append(part)
        print("scene", part.name, f"{duration(part):.1f}s", flush=True)
    end_png(work / "end.png")
    still(work / "end.png", work / "99_end.mp4", 8.0)
    parts.append(work / "99_end.mp4")
    lst = work / "parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    tmp = out.with_name(out.stem + ".part.mp4")
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(tmp)])
    tmp.rename(out)
    print("video:", out, f"{duration(out):.0f}s")


if __name__ == "__main__":
    main()
