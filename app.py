"""
Paint Shop AI Operating System
Main Streamlit Application orchestrating all 7 layers:
L1 Data Foundation | L2 Executive Dashboard | L3 Demand Predictor & Clearance Evaluator
L4 Smart Alerts | L5 Vision & Security | L6 Room Colour AI | L7 AI Shopkeeper Copilot
"""
import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import json
import numpy as np
# NumPy 2.0 compatibility aliases for older packages like xarray
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime

# Local imports
from src.config import (
    STORE_NAME,
    CURRENCY_SYMBOL,
    STORE_OPEN_HOUR,
    STORE_CLOSE_HOUR,
    DEFAULT_CAMERA_INDEX,
    ESP32_DEFAULT_STREAM_URL,
    SAMPLE_ROOMS_DIR
)
from src.database import (
    get_all_products,
    get_inventory_status,
    get_sales_summary,
    get_daily_sales_timeseries,
    get_footfall_pos_fusion,
    get_top_and_slow_shades,
    get_recent_security_events,
    get_seasonal_events
)
from src.dashboard_views import (
    render_sales_trend_chart,
    render_footfall_conversion_chart,
    render_category_donut_chart,
    render_fast_slow_moving_chart,
    render_expiry_timeline_chart
)
from src.predictor import compute_sales_forecast, evaluate_supplier_clearance_deal
from src.smart_alerts import generate_smart_alerts
from src.camera_stream import StoreVisionEngine
from src.colour_assistant import (
    search_shades_by_query,
    find_nearest_shades_by_hex,
    recolor_room_wall,
    hex_to_rgb,
    rgb_to_hex
)
from src.ai_assistant import chat_with_assistant

# --- Page Setup & Modern Styling ---
st.set_page_config(
    page_title=f"{STORE_NAME} | AI Operating System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Bar */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0284C7 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .brand-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 12px;
        color: #10B981;
        font-weight: 600;
        margin-top: 6px;
    }
    .kpi-sub-warn {
        font-size: 12px;
        color: #EF4444;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Alert Cards */
    .alert-card-critical {
        background: #FEF2F2;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .alert-card-warning {
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .alert-card-info {
        background: #F0FDF4;
        border-left: 5px solid #10B981;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .alert-title {
        font-weight: 700;
        font-size: 14px;
        color: #1E293B;
    }
    .alert-desc {
        font-size: 13px;
        color: #475569;
        margin-top: 4px;
    }
    .alert-action {
        font-size: 12px;
        font-weight: 600;
        color: #0369A1;
        margin-top: 6px;
    }

    /* Color Swatch Card */
    .swatch-card {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px;
        background: white;
        margin-bottom: 12px;
    }
    .swatch-box {
        width: 100%;
        height: 48px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #0284C7 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Brand Banner
st.markdown(f"""
<div class="brand-header">
    <div class="brand-title">🎨 {STORE_NAME} <span style="font-size: 14px; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 20px;">AI Operating System v2.0</span></div>
    <div class="brand-subtitle">Real-time Retail Intelligence • Demand Forecasting • Computer Vision • Room Colour AI • Smart Inventory</div>
</div>
""", unsafe_allow_html=True)

# Sidebar System Controls & Status
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1562259949-e8e7689d7828?w=500&auto=format&fit=crop&q=60", use_column_width=True)
    st.title("⚙️ System Control")
    
    # Store Operational Status
    now = datetime.now()
    is_open = (STORE_OPEN_HOUR <= now.hour < STORE_CLOSE_HOUR)
    status_color = "#10B981" if is_open else "#EF4444"
    status_text = "🟢 STORE OPEN" if is_open else "🔴 STORE CLOSED (AFTER-HOURS)"
    
    st.markdown(f"""
    <div style="background:#F8FAFC; border: 1px solid #E2E8F0; padding:12px; border-radius:10px; margin-bottom:15px;">
        <div style="font-size:12px; color:#64748B; font-weight:700;">OPERATING STATUS</div>
        <div style="font-size:15px; font-weight:800; color:{status_color}; margin-top:2px;">{status_text}</div>
        <div style="font-size:12px; color:#64748B; margin-top:4px;">Hours: 09:00 AM – 09:00 PM</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📹 Camera Engine")
    cam_mode = st.radio("Active Source:", ["Laptop / USB Webcam", "ESP32-CAM Stream", "Simulated Camera"], index=0)
    esp32_ip = st.text_input("ESP32 Stream URL:", ESP32_DEFAULT_STREAM_URL) if cam_mode == "ESP32-CAM Stream" else ""
    
    st.subheader("🤖 AI Model Backend")
    api_key_input = st.text_input("Gemini API Key:", value=os.getenv("GEMINI_API_KEY", ""), type="password", help="Leave blank to use the smart local deterministic copilot")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.success("Gemini 2.5 Flash active")
    else:
        st.info("Operating in Offline Deterministic Mode")

    st.markdown("---")
    st.caption("Paint Shop AI Assistant Prototype • 7-Layer Architecture")

# Main Navigation Tabs
tabs = st.tabs([
    "📊 Executive Dashboard",
    "🎨 Room Colour AI",
    "📦 Inventory & Restock",
    "🏷️ Clearance Evaluator",
    "📹 Vision & Security",
    "⚠️ Smart Alerts",
    "🤖 AI Copilot"
])

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD (Layers 1 & 2)
# ==========================================
with tabs[0]:
    sales_summary = get_sales_summary(30)
    fusion_df = get_footfall_pos_fusion(14)
    avg_conv = round(fusion_df["conversion_rate_pct"].mean(), 1) if not fusion_df.empty else 34.5
    
    # 5 KPI Ribbon
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Today's Revenue</div>
            <div class="kpi-value">{CURRENCY_SYMBOL}{sales_summary['today_revenue']:,.0f}</div>
            <div class="kpi-sub">↑ 14% vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Cans Sold Today</div>
            <div class="kpi-value">{sales_summary['today_units']}</div>
            <div class="kpi-sub">{sales_summary['today_invoices']} invoices billed</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Store Footfall Conversion</div>
            <div class="kpi-value">{avg_conv}%</div>
            <div class="kpi-sub">Visitors → Buyers</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Stock Valuation</div>
            <div class="kpi-value">{CURRENCY_SYMBOL}{sales_summary['total_inventory_cost']:,.0f}</div>
            <div class="kpi-sub">{sales_summary['total_active_skus']} active SKUs</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Critical Attention</div>
            <div class="kpi-value" style="color: #EF4444;">{sales_summary['low_stock_count']} Items</div>
            <div class="kpi-sub-warn">Stockout risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Active Seasonal Festival Banner
    festivals = get_seasonal_events()
    st.info(f"🪔 **Active Seasonal Period: {festivals.iloc[0]['event_name']}** — {festivals.iloc[0]['description']} (Expected demand multiplier: **{festivals.iloc[0]['expected_multiplier']}x**)")

    # Row 1 Charts: Sales Trend & Footfall-POS Conversion
    col_chart1, col_chart2 = st.columns([1.1, 0.9])
    with col_chart1:
        daily_sales_df = get_daily_sales_timeseries(45)
        st.plotly_chart(render_sales_trend_chart(daily_sales_df), use_container_width=True)
    with col_chart2:
        st.plotly_chart(render_footfall_conversion_chart(fusion_df), use_container_width=True)

    # Row 2 Charts: Category Valuation & Fast vs Slow Moving
    col_chart3, col_chart4, col_chart5 = st.columns([1, 1, 1])
    top_slow = get_top_and_slow_shades(7)
    inv_df = get_inventory_status()

    with col_chart3:
        st.plotly_chart(render_category_donut_chart(inv_df), use_container_width=True)
    with col_chart4:
        fig_top, _ = render_fast_slow_moving_chart(top_slow["top_shades"], top_slow["slow_shades"])
        st.plotly_chart(fig_top, use_container_width=True)
    with col_chart5:
        _, fig_slow = render_fast_slow_moving_chart(top_slow["top_shades"], top_slow["slow_shades"])
        st.plotly_chart(fig_slow, use_container_width=True)

# ==========================================
# TAB 2: ROOM COLOUR AI (Layer 6)
# ==========================================
with tabs[1]:
    st.subheader("🎨 AI Room Colour Assistant & Visual Recolouring")
    st.markdown("Upload a customer room photo or choose a template, find matching catalog shades, and preview realistic wall recolouring with preserved lighting & shadows.")

    col_room_l, col_room_r = st.columns([1.1, 0.9])

    with col_room_l:
        st.markdown("##### 1. Select Room Image")
        room_source = st.radio("Image Source:", ["Use Demo Room Template", "Upload Custom Room Photo"], horizontal=True)

        room_img = None
        if room_source == "Use Demo Room Template":
            demo_choice = st.selectbox("Select Template:", ["modern_living_room.jpg", "master_bedroom.jpg", "cozy_dining_room.jpg"])
            sample_path = SAMPLE_ROOMS_DIR / demo_choice
            if sample_path.exists():
                room_img = Image.open(sample_path)
        else:
            uploaded_file = st.file_uploader("Upload Room Photo (JPG/PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                room_img = Image.open(uploaded_file)

        if room_img is None:
            # Fallback
            room_img = Image.new("RGB", (600, 400), color=(225, 220, 210))

        st.image(room_img, caption="Original Room Photo", use_column_width=True)

    with col_room_r:
        st.markdown("##### 2. Search & Select Paint Shade")
        color_query = st.text_input("Describe desired vibe / room:", value="warm beige for living room", help="e.g. 'calm blue bedroom', 'terracotta exterior', 'emerald luxury'")
        
        search_results = search_shades_by_query(color_query, limit=4)
        
        selected_shade = None
        if search_results:
            st.markdown("**Matching Paint Shades in Store:**")
            for idx, shade in enumerate(search_results):
                s_col1, s_col2, s_col3 = st.columns([1, 2.5, 1.5])
                with s_col1:
                    st.markdown(f'<div class="swatch-box" style="background:{shade["hex_code"]};"></div>', unsafe_allow_html=True)
                with s_col2:
                    st.markdown(f"**{shade['brand']} {shade['shade_name']}**<br><span style='font-size:12px; color:#64748B;'>Code: {shade['shade_code']} • {shade['finish']}</span>", unsafe_allow_html=True)
                with s_col3:
                    if shade["in_stock"]:
                        st.markdown(f"<span style='color:#10B981; font-weight:700; font-size:12px;'>✓ IN STOCK</span><br><span style='font-size:12px;'>{shade['total_stock']} units</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:#EF4444; font-weight:700; font-size:12px;'>✗ OUT OF STOCK</span>", unsafe_allow_html=True)
                    if st.button("Apply", key=f"btn_apply_{idx}"):
                        selected_shade = shade

            if not selected_shade and search_results:
                selected_shade = search_results[0]
        
        # Color picker fallback
        custom_color = st.color_picker("Or Pick Custom HEX Shade:", value=selected_shade["hex_code"] if selected_shade else "#D9B48C")
        if custom_color and not selected_shade:
            nearest = find_nearest_shades_by_hex(custom_color, limit=1)
            selected_shade = nearest[0] if nearest else None

    # Visual Recolouring Preview Section
    if room_img and selected_shade:
        st.markdown("---")
        st.markdown(f"##### 3. Visual Recolouring Preview: **{selected_shade['brand']} {selected_shade['shade_name']}** (`{selected_shade['hex_code']}`)")
        
        pv_col1, pv_col2 = st.columns([1.1, 0.9])
        with pv_col1:
            # Process wall recolouring
            np_img = np.array(room_img)
            recolored_np = recolor_room_wall(np_img, selected_shade["hex_code"])
            st.image(recolored_np, caption=f"Recoloured Wall Preview with Natural Shadows ({selected_shade['shade_name']})", use_column_width=True)

        with pv_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div style="font-size: 16px; font-weight: 800; color: #0F172A;">Physical Product Specifications</div>
                <div style="margin-top: 10px; font-size: 13px; line-height: 1.8;">
                    <b>Brand:</b> {selected_shade['brand']} ({selected_shade.get('sub_brand', '')})<br>
                    <b>Shade Name:</b> {selected_shade['shade_name']} (Code: <code>{selected_shade['shade_code']}</code>)<br>
                    <b>Category:</b> {selected_shade['category']}<br>
                    <b>Finish:</b> {selected_shade['finish']}<br>
                    <b>Target Hex:</b> <code>{selected_shade['hex_code']}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("###### Available Pack Sizes & Pricing:")
            if selected_shade.get("pack_options"):
                pack_df = pd.DataFrame(selected_shade["pack_options"])
                pack_df = pack_df.rename(columns={"sku": "SKU", "size_litres": "Pack (L)", "price": f"Price ({CURRENCY_SYMBOL})", "stock": "Stock (Cans)"})
                st.dataframe(pack_df, hide_index=True, use_container_width=True)

            if not selected_shade["in_stock"] and selected_shade.get("alternatives"):
                st.warning("⚠️ Requested shade is currently out of stock. Closest in-stock alternatives:")
                for alt in selected_shade["alternatives"]:
                    st.markdown(f"- 🟢 **{alt['brand']} {alt['shade_name']}** (`{alt['shade_code']}`) — Stock: **{alt['current_stock']} cans** | {CURRENCY_SYMBOL}{alt['selling_price']}")

# ==========================================
# TAB 3: INVENTORY & RESTOCK (Layers 1 & 3)
# ==========================================
with tabs[2]:
    st.subheader("📦 Inventory Tracking & Demand Forecasting")
    
    st_filter = st.selectbox("Filter Inventory View:", ["All Products", "Low Stock (Reorder Needed)", "Expiring Soon (<45 Days)", "Overstock / Dead Capital"])
    inv_df = get_inventory_status()

    if st_filter == "Low Stock (Reorder Needed)":
        display_inv = inv_df[inv_df["health_status"] == "LOW_STOCK"]
    elif st_filter == "Expiring Soon (<45 Days)":
        display_inv = inv_df[inv_df["health_status"] == "EXPIRING_SOON"]
    elif st_filter == "Overstock / Dead Capital":
        display_inv = inv_df[inv_df["health_status"] == "OVERSTOCK"]
    else:
        display_inv = inv_df

    st.dataframe(
        display_inv[[
            "sku", "brand", "shade_name", "category", "pack_size_litres",
            "cost_price", "selling_price", "current_stock", "on_order",
            "health_status", "days_to_expiry", "shelf_location", "supplier_name"
        ]].rename(columns={
            "sku": "SKU", "brand": "Brand", "shade_name": "Shade", "category": "Category",
            "pack_size_litres": "Pack", "cost_price": f"Cost ({CURRENCY_SYMBOL})",
            "selling_price": f"Price ({CURRENCY_SYMBOL})", "current_stock": "Stock",
            "on_order": "In-Transit", "health_status": "Status", "days_to_expiry": "Expiry (Days)",
            "shelf_location": "Shelf", "supplier_name": "Supplier"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("📈 Lead-Time Aware Purchase Order Recommendations")
    st.markdown("Demand forecasting incorporates active seasonal festival multipliers (e.g. Diwali whitewash surge), supplier lead times, and Supplier Minimum Order Quantities (MOQ).")

    forecast_df = compute_sales_forecast(30)
    reorder_urgent = forecast_df[forecast_df["suggested_reorder_qty"] > 0]

    st.dataframe(
        reorder_urgent[[
            "sku", "brand", "shade_name", "pack_size_litres", "current_stock",
            "daily_velocity", "seasonal_multiplier", "forecast_30d",
            "days_to_stockout", "lead_time_days", "moq_units",
            "suggested_reorder_qty", "reorder_urgency", "suggested_order_date",
            "estimated_order_cost", "supplier_name"
        ]].rename(columns={
            "sku": "SKU", "brand": "Brand", "shade_name": "Shade", "pack_size_litres": "Pack (L)",
            "current_stock": "Current", "daily_velocity": "Base Velocity/day",
            "seasonal_multiplier": "Festival Mult", "forecast_30d": "30d Demand",
            "days_to_stockout": "Days to Stockout", "lead_time_days": "Lead Time",
            "moq_units": "MOQ", "suggested_reorder_qty": "Order Qty",
            "reorder_urgency": "Urgency", "suggested_order_date": "Order Deadline",
            "estimated_order_cost": f"Cost ({CURRENCY_SYMBOL})", "supplier_name": "Supplier"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# TAB 4: SUPPLIER CLEARANCE EVALUATOR (Layer 3)
# ==========================================
with tabs[3]:
    st.subheader("🏷️ Supplier Clearance Deal Evaluator")
    st.markdown("Evaluate bulk clearance lots offered at heavy discounts by paint distributors. The model balances **profit margin**, **forward velocity**, **holding costs**, and **shelf-life expiry risk** to determine whether to purchase.")

    c_eval1, c_eval2 = st.columns([1, 1.2])

    with c_eval1:
        st.markdown("##### Enter Supplier Offer Parameters")
        products_df = get_all_products()
        prod_options = {f"{r['brand']} {r['shade_name']} ({r['pack_size_litres']}L) - SKU: {r['sku']}": r["id"] for _, r in products_df.iterrows()}
        selected_prod_label = st.selectbox("Select Product Lot:", list(prod_options.keys()))
        selected_prod_id = prod_options[selected_prod_label]

        p_info = products_df[products_df["id"] == selected_prod_id].iloc[0]
        st.caption(f"Standard Cost: {CURRENCY_SYMBOL}{p_info['cost_price']} | Standard Selling Price: {CURRENCY_SYMBOL}{p_info['selling_price']}")

        default_offered = round(float(p_info["cost_price"]) * 0.65, 2)
        offered_cost = st.number_input("Supplier Offered Unit Cost (₹):", min_value=10.0, max_value=float(p_info["selling_price"]), value=default_offered)
        lot_qty = st.number_input("Lot Quantity (Cans):", min_value=5, max_value=1000, value=25, step=5)
        shelf_life_m = st.slider("Remaining Shelf Life (Months):", min_value=1, max_value=36, value=12)

        eval_btn = st.button("Evaluate Deal ROI", type="primary")

    with c_eval2:
        st.markdown("##### Recommendation & Financial Analysis")
        deal_result = evaluate_supplier_clearance_deal(selected_prod_id, offered_cost, lot_qty, shelf_life_m)

        badge_colors = {
            "success": ("#F0FDF4", "#10B981"),
            "warning": ("#FFFBEB", "#F59E0B"),
            "critical": ("#FEF2F2", "#EF4444")
        }
        bg, border = badge_colors.get(deal_result["badge"], ("#F8FAFC", "#64748B"))

        st.markdown(f"""
        <div style="background:{bg}; border-left: 6px solid {border}; padding: 18px; border-radius: 12px; margin-bottom: 20px;">
            <div style="font-size: 18px; font-weight: 800; color: #0F172A;">{deal_result['recommendation']}</div>
            <div style="font-size: 14px; color: #334155; margin-top: 6px;">{deal_result['verdict']}</div>
            <div style="font-size: 13px; font-weight: 700; color: #0369A1; margin-top: 8px;">Recommended Action: {deal_result['action']}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Discount on Cost", f"{deal_result['discount_pct']}%")
        m2.metric("Projected Margin", f"{deal_result['profit_margin_pct']}%")
        m3.metric("Net Projected Profit", f"{CURRENCY_SYMBOL}{deal_result['net_projected_profit']:,.0f}")

        st.caption(f"Estimated time to sell entire lot: **{deal_result['months_to_sell']} months** (Remaining shelf life: {deal_result['shelf_life_months']} months). Total investment required: {CURRENCY_SYMBOL}{deal_result['total_investment']:,.2f}.")

# ==========================================
# TAB 5: VISION & SECURITY (Layer 5)
# ==========================================
with tabs[4]:
    st.subheader("📹 Store Vision, Footfall & After-Hours Security")
    st.markdown("Computer vision engine detecting customer footfall, store occupancy, after-hours motion, and restricted storage zone intrusion.")

    cam_col1, cam_col2 = st.columns([1.2, 0.8])

    source_code = "esp32" if cam_mode == "ESP32-CAM Stream" else "webcam"
    vision_engine = StoreVisionEngine(source_type=source_code, esp32_url=esp32_ip)

    with cam_col1:
        st.markdown("##### Live Camera Stream")
        frame_placeholder = st.empty()
        
        # Capture and process 1 frame
        processed_frame, frame_metrics = vision_engine.process_frame()
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_column_width=True, caption=f"Active Stream ({cam_mode}) • Real-time OpenCV Analysis")

        if st.button("🔄 Refresh Video Frame"):
            st.rerun()

    with cam_col2:
        st.markdown("##### Real-Time Metrics & Security Status")
        
        v_m1, v_m2 = st.columns(2)
        v_m1.metric("People in Frame", frame_metrics["current_people"])
        v_m2.metric("Store Occupancy", frame_metrics["occupancy"])

        if frame_metrics["is_after_hours"]:
            st.error("🚨 STORE CLOSED: After-Hours Security Sentry Armed")
        else:
            st.success("🟢 Operating Hours Active (09:00 AM - 09:00 PM)")

        if frame_metrics.get("security_alert"):
            st.markdown(f"""
            <div class="alert-card-critical">
                <div class="alert-title">🚨 {frame_metrics['security_alert']['type']}</div>
                <div class="alert-desc">{frame_metrics['security_alert']['details']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### Recent Security Event Log")
        sec_df = get_recent_security_events(5)
        st.dataframe(
            sec_df[["timestamp", "event_type", "zone", "severity", "status"]].rename(columns={
                "timestamp": "Time", "event_type": "Event", "zone": "Zone", "severity": "Severity", "status": "Status"
            }),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 6: SMART ALERTS & ANOMALIES (Layer 4)
# ==========================================
with tabs[5]:
    st.subheader("⚠️ Proactive Smart Alerts & Anomaly Center")
    st.markdown("Actionable store intelligence: Low stock warnings, expiring inventory discount promos, dead-capital remediation, and conversion rate drops.")

    all_alerts = generate_smart_alerts()

    al_col1, al_col2 = st.columns(2)

    with al_col1:
        st.markdown(f"##### 🚨 Critical Low Stock ({len(all_alerts['critical'])} Items)")
        for a in all_alerts["critical"]:
            st.markdown(f"""
            <div class="alert-card-critical">
                <div class="alert-title">{a['title']}</div>
                <div class="alert-desc">{a['description']}</div>
                <div class="alert-action">👉 {a['action']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"##### 📉 Traffic Conversion Anomalies ({len(all_alerts['anomalies'])} Events)")
        for a in all_alerts["anomalies"]:
            st.markdown(f"""
            <div class="alert-card-warning">
                <div class="alert-title">{a['title']}</div>
                <div class="alert-desc">{a['description']}</div>
                <div class="alert-action">👉 {a['action']}</div>
            </div>
            """, unsafe_allow_html=True)

    with al_col2:
        st.markdown(f"##### ⏳ Expiring Batches (< 45 Days) ({len(all_alerts['expiring'])} Batches)")
        for a in all_alerts["expiring"]:
            st.markdown(f"""
            <div class="alert-card-warning">
                <div class="alert-title">{a['title']}</div>
                <div class="alert-desc">{a['description']}</div>
                <div class="alert-action">👉 {a['action']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"##### 📦 Dead Capital & Stagnant Stock ({len(all_alerts['dead_stock'])} SKUs)")
        for a in all_alerts["dead_stock"]:
            st.markdown(f"""
            <div class="alert-card-info">
                <div class="alert-title">{a['title']}</div>
                <div class="alert-desc">{a['description']}</div>
                <div class="alert-action">👉 {a['action']}</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 7: AI SHOPKEEPER COPILOT (Layer 7)
# ==========================================
with tabs[6]:
    st.subheader("🤖 Aura — Conversational AI Shopkeeper Copilot")
    st.markdown("Ask natural questions about inventory, demand forecasting, customer color advice, active alerts, or store performance.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello! I am Aura, your **Paint Shop AI Operating System** copilot. I am connected live to your inventory, sales transactions, footfall counters, and color catalogs. How can I assist you today?"}
        ]

    # Quick prompt shortcuts
    st.markdown("**Quick Inquiries:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_input = None
    if q_col1.button("📦 Check Low Stock"):
        quick_input = "What paint products are critically low in stock?"
    if q_col2.button("📈 Reorder Advice"):
        quick_input = "What paints should I order today ahead of Diwali?"
    if q_col3.button("⏳ Expiring Batches"):
        quick_input = "Show me all expiring batches and discount suggestions"
    if q_col4.button("📊 Store KPIs"):
        quick_input = "Give me today's store revenue and footfall conversion"

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_prompt = st.chat_input("Ask Aura anything about your paint shop...") or quick_input

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing live shop data..."):
                response_text = chat_with_assistant(user_prompt, st.session_state.messages)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
