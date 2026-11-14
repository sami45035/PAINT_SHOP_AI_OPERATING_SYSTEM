"""
Film the demo: the real app in real browsers, driven like a person would (a visible
cursor glides to each button), the shop camera and the phone on fake cameras.

Frames are captured from the browser (JPEG, with their own timestamps), then cut
into one clip per scene at 30 fps. Captions, titles and the phone's frame are
added afterwards (tools/edit_demo.py), so wording can change without filming again.

Run:  python tests/browser/record_demo.py OUT_DIR
Needs: streamlit, playwright (chromium), ffmpeg on PATH.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/apex_demo").resolve()

import tests.browser.run_e2e as e2e  # noqa: E402  (same app start-up, fake cameras, barcode stand-in)

e2e.WORK = OUT
LAPTOP = {"width": 1920, "height": 1080}
PHONE = {"width": 540, "height": 1170}     # under the phone breakpoint: the app's mobile layout, crisp in the frame
FPS = 30

CURSOR = """
(() => {
  if (window.__apxCursor) return; window.__apxCursor = 1;
  const touch = %s;
  let c = null, hide = null;
  const mk = () => { const d = document.createElement('div');
    d.style.cssText = 'position:fixed;z-index:2147483647;pointer-events:none;border-radius:50%%;display:none;' +
      (touch ? 'width:46px;height:46px;margin:-23px 0 0 -23px;background:rgba(255,255,255,.55);border:3px solid rgba(11,110,121,.9);'
             : 'width:24px;height:24px;margin:-12px 0 0 -12px;background:rgba(11,110,121,.28);border:2px solid #0B6E79;') +
      'transition:transform .12s ease;';
    document.documentElement.appendChild(d); return d; };
  const at = e => { c = c || mk(); c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; };
  addEventListener('mousemove', e => { at(e); if (!touch) c.style.display = 'block'; }, true);
  addEventListener('mousedown', e => { at(e); c.style.display = 'block'; c.style.transform = 'scale(.72)'; clearTimeout(hide); }, true);
  addEventListener('mouseup', () => { if (!c) return; c.style.transform = 'scale(1)';
    if (touch) hide = setTimeout(() => { c.style.display = 'none'; }, 350); }, true);
  addEventListener('mouseout', e => { if (!e.relatedTarget && c && !touch) c.style.display = 'none'; }, true);
})();
"""


# ----------------------------------------------------------------------------- capture
class Recorder:
    """Browser frames (screencast) to JPEG files with their timestamps; scene marks on the same clock."""

    def __init__(self, ctx, page, name, size):
        self.dir = OUT / f"frames_{name}"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.frames, self.marks, self.name = [], [], name
        self.cdp = ctx.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._on)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                               "maxWidth": size["width"], "maxHeight": size["height"]})

    def _on(self, e):
        ts = float(e["metadata"]["timestamp"])
        path = self.dir / f"{len(self.frames):06d}.jpg"
        import base64
        path.write_bytes(base64.b64decode(e["data"]))
        self.frames.append((ts, path.name))
        try:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": e["sessionId"]})
        except Exception:
            pass

    def mark(self, label):
        self.marks.append((label, time.time()))
        print(f"[{self.name}] {label}", flush=True)

    def stop(self):
        try:
            self.cdp.send("Page.stopScreencast")
        except Exception:
            pass


def cut(rec, start, end, out_path, size):
    """Frames between two marks -> a 30 fps clip (each frame held until the next one arrived)."""
    t0, t1 = dict(rec.marks)[start], dict(rec.marks)[end]
    frames = [(t, n) for t, n in rec.frames if t <= t1]
    before = [f for f in frames if f[0] <= t0]
    inside = [f for f in frames if f[0] > t0]
    seq = ([(t0, before[-1][1])] if before else []) + inside
    if not seq:
        raise RuntimeError(f"no frames for {start}..{end}")
    lst = out_path.with_suffix(".txt")
    with open(lst, "w") as f:
        for (t, n), nxt in zip(seq, seq[1:] + [(t1, None)]):
            f.write(f"file '{rec.dir / n}'\nduration {max(0.001, nxt[0] - t):.4f}\n")
        f.write(f"file '{rec.dir / seq[-1][1]}'\n")
    tmp = out_path.with_name(out_path.stem + ".part.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", f"fps={FPS},scale={size['width']}:{size['height']}:flags=lanczos,format=yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart", str(tmp)],
                   check=True)
    tmp.rename(out_path)
    lst.unlink()


# ----------------------------------------------------------------------------- moving like a person
def glide(page, target, click=True, steps=22, pause=350):
    """Move the cursor smoothly to an element (scrolling it into view gently first), then click."""
    target.scroll_into_view_if_needed()
    page.wait_for_timeout(250)
    box = target.bounding_box()
    if box is None:
        raise RuntimeError("element not on screen")
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y, steps=steps)
    page.wait_for_timeout(pause)
    if click:
        page.mouse.click(x, y)


def scroll(page, pixels, seconds=2.0, x=None, y=None):
    """Smooth wheel scrolling over the main area."""
    vp = page.viewport_size
    page.mouse.move(x or vp["width"] * 0.6, y or vp["height"] * 0.55, steps=8)
    n = max(1, int(seconds * 25))
    for _ in range(n):
        page.mouse.wheel(0, pixels / n)
        page.wait_for_timeout(40)


def nav(page, title):
    glide(page, page.locator('[data-testid="stSidebarNav"] a, [data-testid="stSidebar"] a').filter(has_text=title).first)
    page.wait_for_timeout(400)
    e2e.settle(page, timeout_s=240)


# ----------------------------------------------------------------------------- the film
def film(pw, db):
    vids = OUT / "videos"
    fake = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required",
            "--hide-scrollbars"]
    lb = pw.chromium.launch(args=fake + [f"--use-file-for-fake-video-capture={vids / 'shop.mjpeg'}"])
    pb = pw.chromium.launch(args=fake + [f"--use-file-for-fake-video-capture={vids / 'labels.mjpeg'}"])
    lctx = lb.new_context(viewport=LAPTOP, permissions=["camera"])
    lctx.add_init_script(CURSOR % "false")
    pctx = pb.new_context(viewport=PHONE, is_mobile=True, has_touch=True, device_scale_factor=1, permissions=["camera"],
                          user_agent=pw.devices["Pixel 7"]["user_agent"])
    pctx.expose_function("__apxDecode", e2e.decode)
    pctx.add_init_script(e2e.SHIM)
    pctx.add_init_script(CURSOR % "true")

    # warm-up, not filmed: the wall model downloads on the first Colour Studio visit
    warm = lctx.new_page()
    warm.goto(f"{e2e.BASE}/colour-studio")
    e2e.settle(warm, timeout_s=400)
    warm.close()

    laptop, phone = lctx.new_page(), pctx.new_page()
    L, P = Recorder(lctx, laptop, "laptop", LAPTOP), Recorder(pctx, phone, "phone", PHONE)

    # ---------- the counter: Store Camera on the laptop, Scan & Bill on the phone
    laptop.goto(f"{e2e.BASE}/camera")
    e2e.settle(laptop)
    phone.goto(f"{e2e.BASE}/scan-bill")
    e2e.settle(phone)
    laptop.wait_for_timeout(800)
    L.mark("camera_setup")
    glide(laptop, laptop.locator('[data-testid="stRadio"] label').filter(has_text="Webcam").first)
    e2e.settle(laptop)
    glide(laptop, laptop.get_by_role("tab", name="Set up"))
    laptop.wait_for_timeout(600)
    sec = laptop.get_by_text("Staff uniform — don't count staff as customers")
    sec.scroll_into_view_if_needed()
    glide(laptop, sec, click=False)
    laptop.wait_for_timeout(3500)
    L.mark("camera_setup_end")
    laptop.evaluate("() => window.scrollTo(0, 0)")
    glide(laptop, laptop.get_by_role("tab", name="Live"))
    laptop.wait_for_timeout(500)
    cam = e2e.frame_of(laptop, "browser_cam")
    L.mark("camera_live")
    glide(laptop, cam.locator("#start"))
    t0 = time.time()
    at = lambda s: max(0, s - (time.time() - t0)) * 1000
    laptop.mouse.move(1500, 900, steps=15)               # out of the way of the picture

    phone.wait_for_timeout(at(21))
    L.mark("camera_live_end")                              # A at the counter, staff tagged "not counted"
    P.mark("scan")
    sc = e2e.frame_of(phone, "code_scanner")
    glide(phone, sc.locator("#start"))
    sc.locator(".cart-head .n").filter(has_text="2 cans").wait_for(timeout=30000)
    phone.wait_for_timeout(1200)
    glide(phone, sc.locator('[data-pay="UPI"]'))
    phone.wait_for_timeout(at(38))
    glide(phone, sc.locator("#save"))
    sc.locator(".rcpt").wait_for(timeout=20000)
    phone.wait_for_timeout(1500)
    glide(phone, sc.locator("#phone"))
    sc.locator("#phone").type("98765 43210", delay=90)
    phone.wait_for_timeout(1500)
    P.mark("scan_end")
    glide(phone, sc.locator("#next"))
    phone.wait_for_timeout(700)
    glide(phone, sc.locator("#stop"))
    e2e.settle(phone)

    phone.wait_for_timeout(at(108))
    P.mark("asking")
    glide(phone, phone.get_by_role("button", name="Just asking", exact=True))
    e2e.settle(phone)
    phone.wait_for_timeout(1500)
    P.mark("asking_end")
    L.mark("camera_b")                                    # customer B on the laptop while staff taps "just asking"
    laptop.wait_for_timeout(at(122))
    L.mark("camera_b_end")

    laptop.wait_for_timeout(at(166))
    glide(laptop, cam.locator("#start"))                  # stop the camera before the video loops
    e2e.settle(laptop)
    laptop.reload()
    e2e.settle(laptop)
    e2e.pick_radio(laptop, "Webcam")
    e2e.settle(laptop)
    laptop.wait_for_timeout(800)
    L.mark("buyers")
    glide(laptop, laptop.get_by_role("tab", name="Visitors & buyers"))
    laptop.wait_for_timeout(2500)
    scroll(laptop, 700, 3.0)
    laptop.wait_for_timeout(2500)
    scroll(laptop, 700, 3.0)
    laptop.wait_for_timeout(2500)
    L.mark("buyers_end")

    # ---------- the rest of the app, on the laptop
    laptop.evaluate("() => window.scrollTo(0, 0)")
    nav(laptop, "Overview")
    laptop.wait_for_timeout(600)
    L.mark("overview")
    laptop.wait_for_timeout(2500)
    scroll(laptop, 900, 4.0)
    laptop.wait_for_timeout(1500)
    L.mark("overview_end")

    nav(laptop, "Sales Forecast")
    e2e.charts_drawn(laptop)
    laptop.wait_for_timeout(800)
    L.mark("forecast")
    laptop.wait_for_timeout(2500)
    chart = laptop.locator('[data-testid="stPlotlyChart"]').first
    box = chart.bounding_box()
    if box:
        laptop.mouse.move(box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.5, steps=20)
        for k in range(36):                               # glide along the forecast: the hover shows each day
            laptop.mouse.move(box["x"] + box["width"] * (0.35 + 0.017 * k), box["y"] + box["height"] * 0.5, steps=2)
            laptop.wait_for_timeout(110)
    laptop.wait_for_timeout(800)
    scroll(laptop, 950, 3.5)
    laptop.wait_for_timeout(3500)
    scroll(laptop, 900, 3.5)
    laptop.wait_for_timeout(4000)
    L.mark("forecast_end")

    laptop.evaluate("() => window.scrollTo(0, 0)")
    nav(laptop, "Colour Studio")
    laptop.wait_for_timeout(1000)
    L.mark("colour")
    laptop.wait_for_timeout(3000)
    use = laptop.get_by_role("button", name="Use", exact=True)
    if use.count() > 1:
        glide(laptop, use.nth(1))                         # another suggested shade on the wall
        e2e.settle(laptop, timeout_s=120)
        laptop.wait_for_timeout(2500)
    look = laptop.get_by_role("button", name="Use this look")
    if look.count():
        look.first.scroll_into_view_if_needed()
        laptop.wait_for_timeout(1500)
        glide(laptop, look.first)
        e2e.settle(laptop, timeout_s=120)
        laptop.wait_for_timeout(1500)
        laptop.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
        laptop.wait_for_timeout(3500)
    L.mark("colour_end")

    nav(laptop, "Stock & Reorder")
    laptop.wait_for_timeout(800)
    L.mark("stock")
    laptop.wait_for_timeout(2500)
    scroll(laptop, 800, 3.5)
    laptop.wait_for_timeout(2500)
    L.mark("stock_end")

    laptop.evaluate("() => window.scrollTo(0, 0)")
    nav(laptop, "Alerts")
    laptop.wait_for_timeout(800)
    L.mark("alerts")
    laptop.wait_for_timeout(2000)
    lost = laptop.locator('[data-testid="stRadio"] label').filter(has_text="Lost sales")
    if lost.count():
        glide(laptop, lost.first)
        e2e.settle(laptop)
    laptop.wait_for_timeout(3000)
    L.mark("alerts_end")

    nav(laptop, "Ask Aura")
    laptop.wait_for_timeout(800)
    L.mark("aura")
    box = laptop.locator('[data-testid="stChatInput"] textarea, textarea').first
    glide(laptop, box)
    box.type("Which products should I reorder this week?", delay=55)
    laptop.keyboard.press("Enter")
    e2e.settle(laptop, timeout_s=60)
    laptop.wait_for_timeout(1500)
    scroll(laptop, 500, 2.0)
    laptop.wait_for_timeout(3000)
    L.mark("aura_end")

    L.stop()
    P.stop()
    laptop.wait_for_timeout(500)

    clips = OUT / "clips"
    clips.mkdir(exist_ok=True)
    plan = [(L, "camera_setup", LAPTOP), (L, "camera_live", LAPTOP), (P, "scan", PHONE), (P, "asking", PHONE),
            (L, "camera_b", LAPTOP), (L, "buyers", LAPTOP), (L, "overview", LAPTOP), (L, "forecast", LAPTOP),
            (L, "colour", LAPTOP), (L, "stock", LAPTOP), (L, "alerts", LAPTOP), (L, "aura", LAPTOP)]
    made = []
    for rec, name, size in plan:
        try:
            cut(rec, name, name + "_end", clips / f"{name}.mp4", size)
            made.append(name)
        except Exception as exc:
            print("could not cut", name, exc)
    (OUT / "marks.json").write_text(json.dumps({"laptop": L.marks, "phone": P.marks, "clips": made}, indent=1))
    lb.close()
    pb.close()
    return made


def main():
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "videos" / "shop.mjpeg").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/browser/make_videos.py"), str(OUT / "videos")], check=True)
    db = OUT / "live" / "paint_shop.db"
    shutil.rmtree(db.parent, ignore_errors=True)
    db.parent.mkdir(parents=True)
    shutil.copy(ROOT / "data" / "paint_shop.db", db)
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('camera_layout', ?)",
                  (json.dumps({"count_mode": "presence", "uniform": {"on": True, "hex": e2e.UNIFORM}, "shelves": []}),))
    os.environ["APEX_DEMO_TIME"] = "11:30"          # film an open shop, whatever the time on the filming machine
    proc = e2e.start_app(db)
    try:
        with sync_playwright() as pw:
            made = film(pw, db)
    finally:
        e2e.stop_app(proc)
    for d in OUT.glob("frames_*"):
        shutil.rmtree(d, ignore_errors=True)
    print("clips:", made)


if __name__ == "__main__":
    main()
