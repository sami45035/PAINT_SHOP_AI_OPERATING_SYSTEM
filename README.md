# 🎨 Paint Shop AI Operating System (AI-OS)
> **An AI Operating System for Neighbourhood & Independent Paint Stores**  
> Real-Time Prototype — Complete 7-Layer Architecture

---

## 🌟 Overview & Value Proposition
Small paint shops manage hundreds of shades, brands, pack sizes, seasonal spikes, walk-in customers, and complex purchasing decisions. Many decisions are traditionally made using memory, guesswork, or basic Excel sheets.

The **Paint Shop AI Operating System** unites **Sales Data + Inventory Intelligence + Computer Vision + Machine Learning + Generative AI** into a data-driven operating platform:
- 🟢 **Sell Better**: AI Room Colour Assistant with wall segmentation, lighting-preserved recolouring, and in-stock shade matching.
- 🔵 **Stock Smarter**: Demand forecasting incorporating supplier lead times, MOQs, festival multipliers (e.g. Diwali), and a Supplier Clearance Deal Evaluator.
- 🟠 **Understand the Shop**: Low-cost vision sensors (Webcam / ESP32-CAM) tracking footfall, store conversion rates, peak hours, and event-based security alerts.

---

## 🏛️ 7-Layer Architecture

| Layer | Module | Description | Key Technologies |
|---|---|---|---|
| **Layer 1** | **Shop & Supplier Data** | Relational schema (`products`, `inventory`, `suppliers`, `sales`, `footfall_logs`, `security_events`, `seasonal_calendar`). | SQLite, Pandas |
| **Layer 2** | **Executive Dashboard** | Real-time KPIs, revenue & volume trends, hourly footfall vs. sales conversion rate, category valuation, fast vs. slow-moving stock. | Streamlit, Plotly |
| **Layer 3** | **Demand Predictor & Purchase Engine** | 7-day and 30-day forecasting with festival multipliers (Diwali surge 2.2x), lead-time reordering, and Supplier Clearance Evaluator. | Scikit-learn, NumPy |
| **Layer 4** | **Smart Alerts Center** | Critical low-stock triggers, expiring batch liquidation promos, stagnant dead-capital remediation, and traffic conversion anomaly flags. | Python Rules Engine |
| **Layer 5** | **Vision & Store Security** | Dual-input camera stream (Webcam + ESP32-CAM URL), person detection, directional footfall, after-hours motion sentry, restricted storage zone monitor. | OpenCV, HOG / Cascade |
| **Layer 6** | **AI Room Colour Assistant** | Room photo upload, wall segmentation, photorealistic wall recolouring preserving natural textures/shadows, and physical catalog mapping. | OpenCV LAB Blending, NumPy |
| **Layer 7** | **AI Shopkeeper Copilot (Aura)** | Conversational assistant querying inventory, explaining alerts, evaluating clearance lots, and recommending colors via Gemini (with offline fallback). | Google Gemini, `google-genai` |

---

## 🚀 Quickstart Guide

### 1. Requirements & Dependencies
Ensure Python 3.10+ is installed:
```powershell
pip install -r requirements.txt
```

### 2. Initialize Database & Seed Data
Generate 96+ paint SKUs (Asian Paints, Berger, Dulux, Nerolac), realistic inventory batches, 180 days of sales, hourly footfall, and seasonal events:
```powershell
python data/seed_data.py
```

### 3. Run Automated Master Test Suite
Verify all 7 architectural layers:
```powershell
python -m unittest tests/test_master.py -v
```

### 4. Launch the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📹 ESP32-CAM Hardware Setup (Layer 5)

1. Flash your **ESP32-CAM** with standard CameraWebServer example (or RTSP/MJPEG firmware).
2. Connect the ESP32-CAM to your store's Wi-Fi network.
3. Note the IP address assigned to the board (e.g. `192.168.1.100`).
4. In the application sidebar under **📹 Camera Engine**, select **ESP32-CAM Stream** and paste your stream URL:
   ```
   http://192.168.1.100:81/stream
   ```
5. If the ESP32-CAM is not connected, the application seamlessly defaults to your **Laptop/USB Webcam** or **Simulated Store Camera**.

---

## 🤖 Gemini API Setup (Optional)
To enable live generative reasoning in Layer 7:
```powershell
set GEMINI_API_KEY="your-gemini-api-key"
```
Or paste your key directly into the application sidebar. If no key is provided, the copilot runs in an **intelligent offline deterministic mode** answering all inventory, sales, reorder, and color queries without external network calls.
