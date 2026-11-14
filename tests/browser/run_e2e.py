"""
Browser end-to-end test: the real app (streamlit run) in real browsers with fake cameras.

  1. Every page opens without an error box, "Something went wrong" or "Oh no".
  2. The counter: the laptop runs Store Camera on a video of a scripted shop
     (customer A buys, a staff member in uniform, customer B just asks, a passer-by);
     a phone runs Scan & Bill on a video of two cans' QR labels, saves a UPI bill
     while A is at the counter, sends it on WhatsApp, and later taps "Just asking"
     for B. The screens and the database must say exactly that.

Run:  python tests/browser/run_e2e.py [work_dir]      (makes the videos, starts the app, prints a report)
"""
import base64
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/apex_e2e").resolve()
PORT = 8599
BASE = f"http://localhost:{PORT}"
PAGES = ["overview", "sales-forecast", "colour-studio", "scan-bill", "stock", "alerts", "camera", "ask-aura"]
UNIFORM = "#f57800"

results = []          # (name, ok, detail)


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


# ----------------------------------------------------------------------------- the app
def start_app(db: Path):
    env = dict(os.environ, PAINT_SHOP_DB=str(db), PYTHONUNBUFFERED="1")
    log = open(WORK / "server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(90):
        try:
            if urllib.request.urlopen(f"{BASE}/_stcore/health", timeout=2).read() == b"ok":
                return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("the app didn't start — see server.log")


def stop_app(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(10)
    except Exception:
        pass


# ----------------------------------------------------------------------------- page helpers
def settle(page, quiet_ms=1500, timeout_s=90):
    """Wait until Streamlit has finished running the script (no 'Running…' for a moment)."""
    t0, quiet_since = time.time(), None
    while time.time() - t0 < timeout_s:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0 and \
            "Running" in (page.locator('[data-testid="stStatusWidget"]').first.inner_text() or "")
        if running:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif (time.time() - quiet_since) * 1000 >= quiet_ms:
            return True
        page.wait_for_timeout(250)
    return False


def problems(page):
    """Error boxes, our safety-net messages and Streamlit's crash page."""
    out = []
    n = page.locator('[data-testid="stException"]').count()
    if n:
        out.append(f"{n} exception box(es): " + page.locator('[data-testid="stException"]').first.inner_text()[:300])
    body = page.locator("body").inner_text()
    for words in ("Something went wrong in", "Oh no", "Error running app", "Traceback"):
        if words in body:
            i = body.index(words)
            out.append(body[i:i + 300].replace("\n", " "))
    return out


def frame_of(page, part):
    return page.frame_locator(f'iframe[src*="{part}"]').first


# ----------------------------------------------------------------------------- 1. every page
def page_tour(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    for path in PAGES:
        t0 = time.time()
        page.goto(f"{BASE}/{path}")
        page.wait_for_selector('[data-testid="stMain"], [data-testid="stAppViewContainer"]', timeout=60000)
        done = settle(page)
        took = time.time() - t0
        page.screenshot(path=str(WORK / f"page_{path}.png"), full_page=True)
        bad = problems(page)
        check(f"page {path} opens cleanly", done and not bad, "; ".join(bad) or f"{took:.1f} s")
    # Colour Studio: paint a sample room
    page.goto(f"{BASE}/colour-studio")
    settle(page)
    check("JS errors in the browser during the tour", not errors, "; ".join(errors[:3]))
    ctx.close()


# ----------------------------------------------------------------------------- 2. the counter
DECODER = cv2.QRCodeDetector()


def decode(data_url: str) -> str:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return ""
    text, _, _ = DECODER.detectAndDecode(img)
    return text or ""


# Linux Chromium has no built-in barcode reader; this stand-in reads QR codes with OpenCV,
# so the scanner's own code (beep, repeats, cart, save) is what's tested.
SHIM = """
window.BarcodeDetector = class {
  static async getSupportedFormats() { return ['qr_code']; }
  async detect(v) {
    const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const t = await window.__apxDecode(c.toDataURL('image/jpeg', 0.85));
    return t ? [{ rawValue: t, format: 'qr_code' }] : [];
  }
};
"""


def counter_flow(pw, db: Path):
    vids = WORK / "videos"
    fake = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"]
    laptop_b = pw.chromium.launch(args=fake + [f"--use-file-for-fake-video-capture={vids / 'shop.mjpeg'}"])
    phone_b = pw.chromium.launch(args=fake + [f"--use-file-for-fake-video-capture={vids / 'labels.mjpeg'}"])
    laptop = laptop_b.new_context(viewport={"width": 1440, "height": 1000}, permissions=["camera"]).new_page()
    pctx = phone_b.new_context(**pw.devices["Pixel 7"], permissions=["camera"])
    pctx.expose_function("__apxDecode", decode)
    pctx.add_init_script(SHIM)
    phone = pctx.new_page()
    js_errors = []
    for p in (laptop, phone):
        p.on("pageerror", lambda e: js_errors.append(str(e)))
    popups = []
    pctx.on("page", lambda pg: popups.append(pg.url))

    # stock before
    q = lambda sql, a=(): sqlite3.connect(db).execute(sql, a).fetchall()
    stock = lambda sku: q("SELECT SUM(current_stock) FROM inventory i JOIN products p ON p.id=i.product_id WHERE sku=?", (sku,))[0][0]
    before = {s: stock(s) for s in ("SKU-AS-1001", "SKU-NE-1070")}

    # laptop: Store Camera, webcam -> browser camera
    laptop.goto(f"{BASE}/camera")
    settle(laptop)
    laptop.get_by_text("Webcam", exact=True).click()
    settle(laptop)
    # uniform set-up screen renders with the setting on
    laptop.get_by_role("tab", name="Set up").click()
    laptop.wait_for_timeout(500)
    check("Set up shows the staff-uniform section", laptop.get_by_text("Staff uniform").count() > 0)
    check("Set up has no errors (uniform on)", not problems(laptop), "; ".join(problems(laptop)))
    laptop.get_by_role("tab", name="Live").click()
    cam = frame_of(laptop, "browser_cam")
    cam.locator("#start").click()
    t0 = time.time()
    at = lambda s: max(0, s - (time.time() - t0))

    # phone: Scan & Bill
    phone.goto(f"{BASE}/scan-bill")
    settle(phone)
    check("Scan & Bill opens on the phone", not problems(phone), "; ".join(problems(phone)))
    phone.wait_for_timeout(at(22) * 1000)
    sc = frame_of(phone, "code_scanner")
    sc.locator("#start").click()
    try:
        sc.locator(".cart-head .n").filter(has_text="2 cans").wait_for(timeout=25000)
        check("scanner adds both cans from their QR labels", True, sc.locator(".cart-head").inner_text().replace("\n", " "))
    except Exception:
        check("scanner adds both cans from their QR labels", False, sc.locator("#cart").inner_text()[:200])
    phone.screenshot(path=str(WORK / "phone_cart.png"))
    sc.locator('[data-pay="UPI"]').click()
    phone.wait_for_timeout(at(38) * 1000)
    sc.locator("#save").click()
    try:
        sc.locator(".rcpt").wait_for(timeout=20000)
        txt = sc.locator(".rcpt").inner_text()
        check("after Save: bill shown, camera off", "Bill saved" in txt and "UPI" in txt and
              sc.locator("video").evaluate("v => v.srcObject === null"), txt.split("\n")[0])
    except Exception:
        check("after Save: bill shown, camera off", False, "no receipt")
    phone.screenshot(path=str(WORK / "phone_receipt.png"))
    sc.locator("#phone").fill("98765 43210")
    sc.locator("#wa").click()
    phone.wait_for_timeout(1500)
    check("WhatsApp opens with the bill", any("wa.me/919876543210" in u or "api.whatsapp.com" in u or "whatsapp" in u
                                             for u in popups), popups[-1:] or "no popup")
    sc.locator("#next").click()
    phone.wait_for_timeout(2500)
    check("Next customer: camera on again, bill cleared",
          sc.locator("video").evaluate("v => !!v.srcObject") and "empty" in sc.locator("#cart").inner_text(),
          sc.locator("#state").inner_text())
    sc.locator("#stop").click()
    settle(phone)
    bill = q("SELECT invoice_no, total, payment_mode, customer_phone, status FROM bills WHERE source='app'")
    check("bill saved in the database (UPI, phone kept)", len(bill) == 1 and bill[0][2] == "UPI" and bill[0][3] == "919876543210",
          bill)
    after = {s: stock(s) for s in before}
    check("stock went down by one can each", all(after[s] == before[s] - 1 for s in before), f"{before} -> {after}")

    # B is at the counter 92-135 s: "just asking"
    phone.wait_for_timeout(at(112) * 1000)
    phone.get_by_role("button", name="Just asking / comparing").click()
    settle(phone)
    laptop.screenshot(path=str(WORK / "laptop_live.png"))

    # after the passer-by and the last person leaving
    laptop.wait_for_timeout(at(166) * 1000)
    cam.locator("#start").click()            # stop the camera before the video loops
    settle(laptop)
    laptop.reload()
    settle(laptop)
    laptop.get_by_text("Webcam", exact=True).click()
    settle(laptop)
    body = laptop.locator("body").inner_text()
    laptop.screenshot(path=str(WORK / "laptop_after.png"), full_page=True)
    visits = q("SELECT entered_at, left_at, bill_no, no_sale_reason FROM visits WHERE source='camera' ORDER BY entered_at")
    foot = q("SELECT SUM(count_in), SUM(count_out) FROM footfall_logs WHERE source='camera'")[0]
    check("2 customers counted (staff in uniform and the passer-by not)", foot[0] == 2, f"in/out {foot}, visits {visits}")
    outcome = [(v[2] is not None, v[3]) for v in visits]
    check("customer A bought, customer B didn't (just asking)", outcome == [(True, None), (False, "just_asking")], visits)
    laptop.get_by_role("tab", name="Visitors & buyers").click()
    laptop.wait_for_timeout(800)
    tab = laptop.locator('[data-testid="stTabPanel"]:visible, [role="tabpanel"]:visible').first.inner_text()
    laptop.screenshot(path=str(WORK / "laptop_buyers.png"), full_page=True)
    check("Visitors & buyers screen shows bought / didn't buy", "Bought" in tab and "Didn" in tab, tab[:300].replace("\n", " | "))
    check("no errors on Store Camera after the run", not problems(laptop), "; ".join(problems(laptop)))
    check("no JS errors in either browser", not js_errors, "; ".join(js_errors[:3]))
    laptop_b.close()
    phone_b.close()


def main():
    from playwright.sync_api import sync_playwright

    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "videos" / "shop.mjpeg").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/browser/make_videos.py"), str(WORK / "videos")], check=True)
    db = WORK / "live" / "paint_shop.db"
    shutil.rmtree(db.parent, ignore_errors=True)
    db.parent.mkdir(parents=True)
    # the counter camera's set-up, as the owner would save it: people in view, orange uniform
    shutil.copy(ROOT / "data" / "paint_shop.db", db)
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT)")
        c.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('camera_layout', ?)",
                  (json.dumps({"count_mode": "presence", "uniform": {"on": True, "hex": UNIFORM}, "shelves": []}),))
    proc = start_app(db)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page_tour(browser)
            browser.close()
            counter_flow(pw, db)
    finally:
        stop_app(proc)
    log = (WORK / "server.log").read_text()
    errs = [ln for ln in log.splitlines() if "ERROR" in ln or "Traceback" in ln]
    check("server log has no errors", not errs, "; ".join(errs[:3]))
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
    (WORK / "report.json").write_text(json.dumps(results, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
