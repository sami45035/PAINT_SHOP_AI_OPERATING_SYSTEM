"""
Comprehensive Master Automated Test Suite for Paint Shop AI Operating System
Tests all 7 layers:
- Layer 1: Database schema, products, suppliers, sales & footfall logs
- Layer 2: Dashboard KPIs & Plotly visualization builders
- Layer 3: Demand forecasting, lead-time reordering & supplier clearance deal evaluator
- Layer 4: Smart alerts, expiring batches & dead-stock remediation
- Layer 5: Camera vision, person detection & security rules
- Layer 6: Color search, Delta-E distance & wall recolouring
- Layer 7: AI Copilot tools & deterministic reasoning fallback
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DB_PATH, SHADES_CATALOG_PATH
from src.database import (
    get_all_products,
    get_inventory_status,
    get_sales_summary,
    get_footfall_pos_fusion,
    get_top_and_slow_shades,
    get_recent_security_events
)
from src.predictor import compute_sales_forecast, evaluate_supplier_clearance_deal
from src.smart_alerts import generate_smart_alerts
from src.colour_assistant import (
    search_shades_by_query,
    find_nearest_shades_by_hex,
    recolor_room_wall,
    segment_wall_mask
)
from src.camera_stream import StoreVisionEngine
from src.ai_assistant import deterministic_assistant_response

class TestPaintShopAIOS(unittest.TestCase):

    def test_01_database_and_catalog(self):
        """Layer 1: Verify database tables, product counts, and inventory joins."""
        self.assertTrue(DB_PATH.exists(), "paint_shop.db must exist")
        self.assertTrue(SHADES_CATALOG_PATH.exists(), "shades_catalog.json must exist")

        products_df = get_all_products()
        self.assertGreaterEqual(len(products_df), 90, "Should have >= 90 products in catalog")
        self.assertIn("supplier_name", products_df.columns)
        self.assertIn("lead_time_days", products_df.columns)

        inv_df = get_inventory_status()
        self.assertGreaterEqual(len(inv_df), 90, "Inventory records should match products")
        self.assertIn("health_status", inv_df.columns)
        self.assertIn("days_to_expiry", inv_df.columns)

    def test_02_dashboard_kpis_and_fusion(self):
        """Layer 2: Verify executive metrics and footfall-POS conversion fusion."""
        summary = get_sales_summary(30)
        self.assertGreater(summary["today_revenue"], 0)
        self.assertGreater(summary["total_inventory_cost"], 10000)
        self.assertGreater(summary["total_active_skus"], 50)

        fusion_df = get_footfall_pos_fusion(7)
        self.assertFalse(fusion_df.empty, "Fusion dataframe should contain merged records")
        self.assertIn("conversion_rate_pct", fusion_df.columns)
        self.assertIn("atv", fusion_df.columns)

        top_slow = get_top_and_slow_shades(5)
        self.assertIn("top_shades", top_slow)
        self.assertIn("slow_shades", top_slow)

    def test_03_sales_predictor_and_reorder(self):
        """Layer 3: Verify 7d/30d forecasting and lead-time reordering."""
        forecast_df = compute_sales_forecast(30)
        self.assertFalse(forecast_df.empty)
        self.assertIn("forecast_7d", forecast_df.columns)
        self.assertIn("forecast_30d", forecast_df.columns)
        self.assertIn("suggested_reorder_qty", forecast_df.columns)
        self.assertIn("reorder_urgency", forecast_df.columns)
        self.assertIn("suggested_order_date", forecast_df.columns)

    def test_04_supplier_clearance_evaluator(self):
        """Layer 3: Verify supplier clearance deal evaluation logic."""
        # Pick product with solid sales (Product 1: Royale Blue 1L)
        prod_id = 1
        reg_cost = 450.0

        # Test A: Lucrative deal (high discount, good demand, long shelf life)
        res_good = evaluate_supplier_clearance_deal(
            product_id=prod_id,
            offered_unit_cost=reg_cost * 0.5,  # 50% discount
            lot_quantity=10,
            shelf_life_months=24
        )
        self.assertIn("recommendation", res_good)
        self.assertIn("net_projected_profit", res_good)
        self.assertIn(res_good["badge"], ["success", "warning"])

        # Test B: Excessive lot with imminent expiry
        res_bad = evaluate_supplier_clearance_deal(
            product_id=prod_id,
            offered_unit_cost=reg_cost * 0.9,
            lot_quantity=600,
            shelf_life_months=2
        )
        self.assertIn("REJECT", res_bad["recommendation"])
        self.assertEqual(res_bad["badge"], "critical")

    def test_05_smart_alerts_engine(self):
        """Layer 4: Verify critical low-stock, expiry, and dead-capital alerts."""
        alerts = generate_smart_alerts()
        self.assertIn("critical", alerts)
        self.assertIn("expiring", alerts)
        self.assertIn("dead_stock", alerts)
        self.assertIn("anomalies", alerts)

        self.assertGreater(len(alerts["critical"]), 0, "Should detect low stock items")
        self.assertGreater(len(alerts["expiring"]), 0, "Should detect expiring batches")

    def test_06_camera_vision_and_security(self):
        """Layer 5: Verify camera frame processing and security monitoring."""
        engine = StoreVisionEngine(source_type="webcam")
        frame, metrics = engine.process_frame()
        self.assertEqual(frame.shape[0], 480)
        self.assertEqual(frame.shape[1], 640)
        self.assertIn("occupancy", metrics)
        self.assertIn("is_after_hours", metrics)

    def test_07_colour_assistant_and_recolouring(self):
        """Layer 6: Verify shade search, stock enrichment, and wall recolouring."""
        matches = search_shades_by_query("blue living room", limit=3)
        self.assertGreater(len(matches), 0)
        self.assertIn("pack_options", matches[0])
        self.assertIn("total_stock", matches[0])

        hex_matches = find_nearest_shades_by_hex("#D9B48C", limit=3)
        self.assertEqual(len(hex_matches), 3)

        # Test wall recolouring on a test image
        test_img = np.full((300, 400, 3), (220, 220, 220), dtype=np.uint8)
        recolored = recolor_room_wall(test_img, "#1A365D")
        self.assertEqual(recolored.shape, test_img.shape)

    def test_08_ai_copilot_deterministic_engine(self):
        """Layer 7: Verify offline assistant handles key store queries."""
        resp_stock = deterministic_assistant_response("Do we have Royale Blue in stock?")
        self.assertIn("Royale Blue", resp_stock)

        resp_order = deterministic_assistant_response("What paint should I reorder today?")
        self.assertTrue("Reorder" in resp_order or "healthy" in resp_order)

        resp_kpi = deterministic_assistant_response("What is today's store revenue and footfall?")
        self.assertIn("Store Performance Snapshot", resp_kpi)

if __name__ == "__main__":
    unittest.main()
