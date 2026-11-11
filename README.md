# Paint Shop AI — Apex Colors & Paints Studio
> An AI assistant for independent paint shops: stock the right paint, stop waste and theft, and serve customers faster.
> Real-time prototype built layer by layer.

---

## What's in it

| Screen | Question it answers | Layer |
|---|---|---|
| **Overview** | How is the shop doing today? | L1 data · L2 dashboard |
| **Colour Studio** | How will this shade look on the customer's wall? | L6 |
| **Stock & Reorder** (+ Deal Checker) | What should I order, how much, and by when? Is this supplier offer worth it? | L3 |
| **Alerts** | What needs my attention right now? | L4 |
| **Store Camera** | Who came in, and is the shop safe after closing? | L5 |
| **Ask Aura** | Ask anything about the shop in plain words | L7 |

## Architecture

| Layer | Module | How it works |
|---|---|---|
| 1 · Shop data | `src/database.py`, `data/seed_data.py` | SQLite: products, inventory batches, suppliers, sales, hourly visitors, security events, season calendar. All dates come from `src/clock.py` (store time zone, India by default). |
| 2 · Overview | `src/ui/pages/overview.py`, `src/dashboard_views.py` | KPIs, sales trend, visitors vs bills by hour. Conversion = bills ÷ visitors for the day. |
| 3 · Forecast & reorder | `src/predictor.py` | Seasonally-adjusted 30-day moving average: the season effect is removed from past sales, then re-applied day by day going forward (a festival ending on 15 Nov only boosts demand until 15 Nov). Order quantity respects supplier lead time and minimum order. |
| 4 · Alerts | `src/smart_alerts.py` | Running out before the supplier can deliver, batches expiring within 45 days, money stuck in slow sellers, days with many visitors but few bills. |
| 5 · Store Camera | `src/camera_stream.py` | Laptop webcam or ESP32-CAM. HOG person detection, line-crossing door counts, after-hours motion guard and staff-only zone, one saved alert per minute with a photo. A clearly-labelled demo feed that never writes to the database. |
| 6 · Colour Studio | `src/colour_studio/` | SegFormer (ADE20K) via ONNX Runtime finds walls and separates them from ceiling, doors, windows and furniture; walls are split at corners; light-preserving recolouring in linear RGB; room-aware shade suggestions; paint quantity planner. |
| 7 · Ask Aura | `src/ai_assistant.py` | Gemini (optional) grounded in live shop data, with an offline mode for common questions. |

UI system (names, colours, components) lives in `src/ui/theme.py` and `.streamlit/config.toml`.

---

## Quick start

Python 3.10+.

```powershell
pip install -r requirements.txt
python data/seed_data.py          # sample data ending today
python -m pytest -q               # or: python -m unittest discover tests
streamlit run app.py
```

Open `http://localhost:8501`.

### Demo mode
The sample data is moved forward automatically so its last day is always "today" (expiry dates move with it). When the shop runs on real data, turn this off:

```powershell
set PAINT_SHOP_DEMO_MODE=0
```

---

## Colour Studio: the wall model

On first use, Colour Studio downloads the wall-finding model once (SegFormer-B3 trained on ADE20K, ONNX format, from Hugging Face: `Xenova/segformer-b3-finetuned-ade-512-512`) into `models/`. It runs on the CPU — no GPU needed.

- **Faster but less accurate:** `set WALL_MODEL_REPO=Xenova/segformer-b0-finetuned-ade-512-512`
- **Use a model file you already have:** `set WALL_MODEL_PATH=C:\path\to\model.onnx` (put its `config.json` next to it)
- **No internet / download failed:** Colour Studio switches to **tap mode** — tap each wall on the photo and it grows the wall from that point, stopping at corners and edges.
- **Faster wall finding:** `set WALL_MODEL_FLIP_PASS=0` skips the second (mirrored) model pass — about half the time, slightly rougher edges.

### How walls are found (`src/colour_studio/refine.py`)
The model predicts on a coarse grid, so its raw wall map has holes at lamp glow, leaks onto headboards and sofas, and ragged edges. The clean-up uses the photo itself:
1. **Blotches** with no visible outline and the wall's colour are wall (a real frame or curtain has an outline).
2. **Leaks** cut off from the wall by an edge but blending into furniture are removed.
3. **Edges** — every border pixel is re-decided so the border sits on the strongest nearby edge.
4. **Growth** across smooth colour changes fills bites (a light pool fading out is smooth; a ceiling corner isn't).
5. **Holes** that aren't objects are filled.
6. **Ceiling line** — a wall meets the ceiling in a straight line; where glare hides it, paint goes up to the fitted line.

Walls are then split at corners and named from perspective (flat ceiling line = back wall, sloping = left/right).

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

## Store Camera setup

- **Webcam:** when the app runs on your laptop, the server reads the webcam directly (about 8 frames a second). On Streamlit Cloud the server can't reach your camera, so the screen switches to **Browser camera**: press **Start camera**, allow access, and the browser shows the live video and sends a small still about twice a second for counting. Works on a laptop or phone (pick the camera from the list if there are several). Keep the tab open while counting.
- **ESP32-CAM:** flash the CameraWebServer example, join the shop Wi-Fi, then put the stream address (e.g. `http://192.168.1.100:81/stream`) in **Settings** in the sidebar.
- Door line and staff-only zone can be adjusted on the Store Camera screen. Put the door line where people's **feet** cross the doorway; the camera should see whole people walking through.
- The camera is read continuously in the background (about 8 frames a second) while the app is open; the screen refreshes every 2 seconds. A person is counted once going in and once going out — standing on the line doesn't add counts, and someone close to the camera with their feet out of view isn't counted (they still show under **In view**).
- **Came in / Left** on this screen are what the camera counted today. The sample visitor numbers on Overview are separate.
- Alert photos are saved to `data/security_snapshots/`.

## Gemini (optional)

```powershell
set GEMINI_API_KEY=your-key
```
or paste it under **Settings** in the sidebar, then switch Ask Aura to **Gemini**. Without a key Aura answers offline.
