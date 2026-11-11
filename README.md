# Paint Shop AI — Apex Colors & Paints Studio
> An AI assistant for independent paint shops: stock the right paint, stop waste and theft, and serve customers faster.
> Real-time prototype built layer by layer.

---

## What's in it

| Screen | Question it answers | Layer |
|---|---|---|
| **Overview** | How is the shop doing today? | L1 data · L2 dashboard |
| **Sales Forecast** | What will sell in the next 30 days — and can I trust the forecast? | L3 |
| **Colour Studio** | How will this shade look on the customer's wall? | L6 |
| **Stock & Reorder** (+ Deal Checker) | What should I order, how much, and by when? Is this supplier offer worth it? | L3 |
| **Alerts** | What needs my attention right now? | L4 |
| **Store Camera** | How many came in, how many bought, what's on the shelves, what moves fast, did anything leave unpaid? | L5 |
| **Ask Aura** | Ask anything about the shop in plain words | L7 |

## Architecture

| Layer | Module | How it works |
|---|---|---|
| 1 · Shop data | `src/database.py`, `data/seed_data.py` | SQLite: products, inventory batches, suppliers, sales, hourly visitors, security events, season calendar. All dates come from `src/clock.py` (store time zone, India by default). |
| 2 · Overview | `src/ui/pages/overview.py`, `src/dashboard_views.py` | KPIs, sales trend, visitors vs bills by hour. Conversion = bills ÷ visitors for the day. |
| 3 · AI sales forecast & reorder | `src/sales_forecast.py`, `src/predictor.py` | Per product group, a ridge regression on six months of daily sales learns the weekday pattern, the real effect of each festival in the calendar (the calendar's multiplier is only a starting guess) and the trend; forecasts are split across products by recent share. Backtested against the simple 30-day average on periods it never saw. Reorder quantities and dates use the same forecast, with supplier lead time and minimum order. |
| 4 · Alerts | `src/smart_alerts.py` | Running out before the supplier can deliver, batches expiring within 45 days, money stuck in slow sellers, days with many visitors but few bills. |
| 5 · Store Camera | `src/camera_stream.py`, `src/shop_vision.py`, `src/shop_insights.py` | The camera as a data source. People detection (HOG) and tracking; door-line footfall; each customer's journey (shelves visited, cans taken / put back, reached the counter); can counting on shelf slots; walk-out (unpaid can) alerts; insights that join all of it with sales, stock and the forecast. |
| 6 · Colour Studio | `src/colour_studio/` | SegFormer (ADE20K) via ONNX Runtime finds walls and separates them from ceiling, doors, windows and furniture; walls are split at corners; light-preserving recolouring in linear RGB; in-browser brush touch-ups; **designer looks** (whole 60-30-10 schemes scored against the room) shown on the customer's own walls; paint quantity planner. |
| 7 · Ask Aura | `src/ai_assistant.py` | Gemini (optional) grounded in live shop data, with an offline mode for common questions. |

UI system (names, colours, components) lives in `src/ui/theme.py` and `.streamlit/config.toml`.

---

## Quick start

Python 3.10+.

```powershell
pip install -r requirements.txt
python data/seed_data.py          # sample data ending today (data/paint_shop.db)
python -m pytest -q               # or: python -m unittest discover tests
streamlit run app.py
```

Open `http://localhost:8501`.

### Where the data lives
`data/paint_shop.db` is the **sample** database (in git). The app works on a **live copy** in `data/live/` (not in git), made from the sample on first start — so updating the code (or a redeploy on Streamlit Cloud) never overwrites the database under a running app or wipes what a camera recorded. To start fresh, delete `data/live/`. In demo mode, a newly seeded sample replaces the live copy automatically. Set `PAINT_SHOP_DB` to use a different file.

### If something goes wrong
Every screen runs inside a safety net (`src/safety.py`): an error shows a message on that screen with its details instead of taking the whole app down ("Oh no" on Streamlit Cloud). The last problems are listed under **Settings** in the sidebar and written to `data/live/errors.log` and the server log (lines starting `[apex] ERROR`).

### Demo mode
The sample data is moved forward automatically so its last day is always "today" (expiry dates move with it). When the shop runs on real data, turn this off:

```powershell
set PAINT_SHOP_DEMO_MODE=0
```

---

## Sales Forecast

The **Sales Forecast** screen shows the next 30 days (rupees or cans) with a likely range, festival periods, what will sell most, what the model learned (calendar guess vs your sales: e.g. Diwali 2.2× guessed, about 1.7× real for interior emulsion), the weekday pattern, a per-product view, and an honest accuracy test:

- **Backtest:** fit only on sales before a cut-off, forecast the next 4 weeks without seeing them, compare with what sold — and with the usual shop method (average of the last 30 days).
- **Repeated test:** the same over the last 10 two-week and four-week periods. On the sample data the AI forecast's totals are off by about 5–6% on average for two weeks (simple average: about 12%), closer in most periods.
- A festival the history hasn't seen yet counts at half the calendar's guess until the shop's own sales confirm it.

In demo mode the sample sales, camera history and the sample festival calendar move forward together, so the forecast keeps learning from aligned data.

## Colour Studio: the wall model

On first use, Colour Studio downloads the wall-finding model once (SegFormer-B3 trained on ADE20K, ONNX format, from Hugging Face: `Xenova/segformer-b3-finetuned-ade-512-512`) into `models/`. It runs on the CPU — no GPU needed.

- **Faster but less accurate:** `set WALL_MODEL_REPO=Xenova/segformer-b0-finetuned-ade-512-512`
- **Use a model file you already have:** `set WALL_MODEL_PATH=C:\path\to\model.onnx` (put its `config.json` next to it)
- **No internet / download failed:** Colour Studio switches to **tap mode** — tap each wall on the photo and it grows the wall from that point, stopping at corners and edges.
- **Faster wall finding:** `set WALL_MODEL_FLIP_PASS=0` skips the second (mirrored) model pass — about half the time, slightly rougher edges.
- **Memory:** the model runs without ONNX Runtime's memory pool, on 2 threads, and frees memory after each photo. On Streamlit Cloud (about 1 GB per app) it sees photos at 448 px on the short side (768 max) instead of 512/896. Override with `WALL_MODEL_INPUT_SIZE`, `WALL_MODEL_MAX_LONG_SIDE`, `WALL_MODEL_THREADS`. The server log shows `[apex] memory …` lines around each run. If an app host has very little memory, `WALL_MODEL_REPO=Xenova/segformer-b0-finetuned-ade-512-512` uses a much lighter model.

### How walls are found (`src/colour_studio/refine.py`)
The model predicts on a coarse grid, so its raw wall map has holes at lamp glow, leaks onto headboards and sofas, and ragged edges. The clean-up uses the photo itself:
1. **Blotches** with no visible outline and the wall's colour are wall (a real frame or curtain has an outline).
2. **Leaks** cut off from the wall by an edge but blending into furniture are removed.
3. **Edges** — every border pixel is re-decided so the border sits on the strongest nearby edge.
4. **Growth** across smooth colour changes fills bites (a light pool fading out is smooth; a ceiling corner isn't).
5. **Holes** that aren't objects are filled.
6. **Ceiling line** — a wall meets the ceiling in a straight line; where glare hides it, paint goes up to the fitted line.

Walls are then split at corners and named from perspective (flat ceiling line = back wall, sloping = left/right).

### Designer looks (colour prediction)
Under the photo, **Designer looks for this room** shows six complete schemes painted on the customer's own room — walls (60%), a feature wall (30%), trim & ceiling (10%) and two decor accents — like a Pinterest board.
- The room is read from the photo: room type (bed → bedroom, sofa → living room…), undertone (wooden floor and furniture → warm), light (a dim room gets lighter walls), and the big furniture (a feature wall the colour of the sofa is marked down; one that echoes the bedding or plays off the sofa is marked up).
- Style chips (Modern, Japandi, Boho, Indian festive, Luxe…) and mood words (*cozy*, *calm*, *festive*) steer it.
- **Use this look** paints the walls (feature wall = back wall; change it in the look panel). Every paint is matched to a shade the shop stocks when one is close (ΔE < 6); otherwise it's an *Apex House Range* colour tinted at the counter (sample shade card with invented APX codes — replace with the shop's real one).
- **Save look board** downloads a Pinterest-style card: the room in the look, swatches, shade names and codes.
- Looks live in `src/colour_studio/palettes.py`.

### Touching up by hand
Switch **Pick a wall** to **Touch up with brush**. The brush runs in the browser, so paint appears as you move:
- **Paint** covers patches the wall finder missed (lamp glare, corners); **Erase** takes paint off the headboard, ceiling or lamps.
- The size slider (or `[` / `]`, or Alt + scroll) sets the brush size; the round cursor shows exactly what it will cover.
- **Stop at edges** only paints pixels that look like the spot where the stroke started — handy along a headboard or frame.
- **Undo** (Ctrl+Z), **Clear**, and hold **👁 Compare** to see the original photo.
Touch-ups stay when you change the shade or the walls, and are used for the shade ideas, the paint amount and the saved before/after.

### If a photo still comes out wrong
```powershell
python tools/wall_debug.py path\to\room.jpg
```
It saves `room_walls.jpg` (model / cleaned / named walls side by side) and a small `room_walls.npz`. Send those with the photo so the clean-up can be tuned on what the real model sees.

Licence note: NVIDIA's SegFormer weights are released for research / non-commercial use. Fine for this prototype; check the licence (or swap the model — it sits behind one class, `WallSegmenter`) before commercial use.

To redraw the three sample rooms: `python data/make_sample_rooms.py`.

---

## Store Camera

The camera answers five questions, each joined with the till and the stock book:

| Question | How | Where |
|---|---|---|
| **Footfall** — how many came in, when? | Feet crossing the door line (downward = in). Only crossings through the doorway count. | Visitors & buyers |
| **Buyers** — how many actually bought? | Bills ÷ visitors; each visit is followed to see if it reached the billing counter. Hours with many visitors and few bills are flagged with the sales they cost. | Visitors & buyers |
| **Theft** | A can leaves a shelf while someone stands there, and that person leaves without reaching the counter → *Left with an unpaid can* alert with a photo. Cans taken off shelves are also checked against the day's bills (*shelf vs till*). Plus after-hours movement and customers in the staff-only area. | Theft & safety, Alerts |
| **Fast movers** | Fastest sellers from the till, plus for watched shelves how often each can is picked up and put back. *Picked up, not bought* and *demand building* are flagged. | Shelves & fast movers |
| **Shelf stock → pre-order** | Cans counted on each shelf slot from the picture. Low shelf + cans in the store room → *Refill now*. The forecast decides what to order; for watched products the camera's pick-up trend sizes the order up or down. | Shelves & fast movers, Alerts |

**Data sources.** Rows are tagged `camera` (a real camera), `demo` (the drawn demo shop) or `sample` (30 days of history built to match the sample sales, so the screens show something on day one). Demo numbers never count as real visitors. `python data/seed_camera.py` adds the sample history to an older database.

**Demo shop.** A drawn shop on a 64-second loop, read by the same code as a real camera: a buyer, a browser who puts a can back, a walk-out (alert), an enquiry, and staff refilling shelves from the store room (staff aren't counted as visitors).

**Set up a real camera** (Set up tab): tap two points for the door line, two corners for the billing counter and the staff-only area, and two corners around each shelf slot (one product per slot, cans side by side facing the camera). Save, then **Calibrate** with the shelves as they are.

**Where the camera is read**
- **Webcam, app running on your laptop:** the app reads it directly (~8 frames a second).
- **Webcam, app on Streamlit Cloud:** the browser streams it — press **Start camera** and allow access (~2 frames a second; keep the tab open).
- **ESP32-CAM:** flash the CameraWebServer example, join the shop Wi-Fi, put the stream address (e.g. `http://192.168.1.100:81/stream`) in **Settings**. The ESP32 must be reachable from where the app runs (so run the app on a laptop or mini PC in the shop).

Alert photos are saved to `data/security_snapshots/`.

## Gemini (optional)

```powershell
set GEMINI_API_KEY=your-key
```
or paste it under **Settings** in the sidebar, then switch Ask Aura to **Gemini**. Without a key Aura answers offline.
