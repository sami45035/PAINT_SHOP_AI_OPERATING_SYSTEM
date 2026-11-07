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

Licence note: NVIDIA's SegFormer weights are released for research / non-commercial use. Fine for this prototype; check the licence (or swap the model — it sits behind one class, `WallSegmenter`) before commercial use.

To redraw the three sample rooms: `python data/make_sample_rooms.py`.

---

## Store Camera setup

- **Laptop webcam:** works when the app runs on that laptop. On Streamlit Cloud the server can't see your laptop's camera — use the browser-camera snapshot on the same screen instead.
- **ESP32-CAM:** flash the CameraWebServer example, join the shop Wi-Fi, then put the stream address (e.g. `http://192.168.1.100:81/stream`) in **Settings** in the sidebar.
- Door line and staff-only zone can be adjusted on the Store Camera screen.
- Alert photos are saved to `data/security_snapshots/`.

## Gemini (optional)

```powershell
set GEMINI_API_KEY=your-key
```
or paste it under **Settings** in the sidebar, then switch Ask Aura to **Gemini**. Without a key Aura answers offline.
