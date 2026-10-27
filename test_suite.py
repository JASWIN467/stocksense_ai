"""
StockSense AI - Master Independent Feature Test Suite
Systematic, automated verification of all 11 features and backend APIs:
1. Forecast Summary Dashboard
2. Product-Level Demand Forecast & Confidence Intervals
3. CSV Validation & Batch Forecasting (Valid, Missing Columns, Empty, Bad Extension)
4. Forecast History Persistence (SQLite CRUD, Pagination, Isolation)
5. Inventory Recommendation & Operations Research Logic (SS, ROP, EOQ, Position)
6. Demand by Category & Market Share Aggregations
7. Sales Trend & Seasonality Analysis
8. Price & Promotion Impact / What-If Simulation
9. Stockout Risk Alerts & Urgency Scoring
10. Model Benchmarking & Explainable AI (XAI)
11. Reports & Export Hub (PDF & CSV verification)
"""

import os
import io
import json
import unittest
import numpy as np
import pandas as pd
import joblib

import app
from inventory_engine import InventoryOptimizer, norm_cdf, z_score_from_service_level
from inference import get_engine
from analytics_engine import get_analytics
from history_db import init_db, save_forecast_run, get_history_runs, delete_run, clear_history
import pdf_generator


class MasterFeatureTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()
        cls.engine = get_engine()
        cls.analytics = get_analytics()
        cls.optimizer = InventoryOptimizer(default_lead_time_days=3, service_level=0.95)

    # =========================================================================
    # FEATURE 1: FORECAST SUMMARY DASHBOARD
    # =========================================================================
    def test_feature1_dashboard_history_and_forecast_continuity(self):
        """Verifies 30-day historical data + 7-day future projection continuity and CI."""
        res = self.client.get('/api/history?store_id=S001&product_id=P0001&model_choice=champion')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        
        hist = data["historical"]
        fc = data["forecast"]
        stats = data["stats"]

        # Check lengths
        self.assertEqual(len(hist["dates"]), 30, "Must return exactly 30 days of history")
        self.assertEqual(len(hist["units_sold"]), 30)
        self.assertEqual(len(hist["inventory_level"]), 30)
        self.assertEqual(len(fc["dates"]), 7, "Must return exactly 7 forecast horizon days")
        self.assertEqual(len(fc["units_sold"]), 7)

        # Check finite non-negative values
        for u in hist["units_sold"]:
            self.assertGreaterEqual(u, 0)
        for u in fc["units_sold"]:
            self.assertGreaterEqual(u, 0)
        for lo, hi, val in zip(fc["ci_lower"], fc["ci_upper"], fc["units_sold"]):
            self.assertLessEqual(lo, val, "CI lower bound must be <= predicted value")
            self.assertGreaterEqual(hi, val, "CI upper bound must be >= predicted value")

        # Verify summary stats match underlying arrays
        calc_mean = round(float(np.mean(hist["units_sold"])), 1)
        self.assertEqual(stats["mean_30d"], calc_mean)

    def test_feature1_dashboard_invalid_store_or_product(self):
        """Dashboard handles unrecognised SKU gracefully without crashing."""
        res = self.client.get('/api/history?store_id=NONEXISTENT&product_id=NONEXISTENT')
        # Empty historical lookup returns 404 or empty history gracefully
        self.assertIn(res.status_code, [200, 404])

    # =========================================================================
    # FEATURE 2: PRODUCT-LEVEL DEMAND FORECAST
    # =========================================================================
    def test_feature2_product_forecast_multi_model(self):
        """Tests demand prediction across Champion (LGBM), Baseline (LR), and RF."""
        models = ["champion", "baseline_lr", "rf"]
        predictions = {}

        for m in models:
            payload = {
                "store_id": "S001",
                "product_id": "P0001",
                "price": 45.0,
                "discount": 10,
                "model_choice": m
            }
            res = self.client.post('/api/predict', json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.get_json()["data"]
            u = data["forecast"]["predicted_units_sold"]
            self.assertGreater(u, 0, f"Model {m} must predict positive units")
            predictions[m] = u

            # Check confidence intervals
            ci = data["forecast"]["confidence_interval_95"]
            self.assertLessEqual(ci["lower"], u)
            self.assertGreaterEqual(ci["upper"], u)

            # Check feature contributions
            self.assertIn("feature_contributions", data)
            self.assertGreater(len(data["feature_contributions"]), 0)

        # Verify models yield reasonable, finite predictions
        self.assertTrue(all(np.isfinite(v) for v in predictions.values()))

    def test_feature2_product_forecast_edge_cases(self):
        """Verifies boundary conditions: negative price, extreme discount, zero stock."""
        # Negative price should be clamped to >= 1.0
        payload_neg_price = {"store_id": "S001", "product_id": "P0001", "price": -50.0}
        res1 = self.client.post('/api/predict', json=payload_neg_price)
        self.assertEqual(res1.status_code, 200)
        self.assertGreaterEqual(res1.get_json()["data"]["features_applied"]["price"], 1.0)

        # Extreme discount > 100% should be clamped to 100%
        payload_huge_disc = {"store_id": "S001", "product_id": "P0001", "discount": 150}
        res2 = self.client.post('/api/predict', json=payload_huge_disc)
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()["data"]["features_applied"]["discount_pct"], 100)

        # Zero stock
        payload_zero_stock = {"store_id": "S001", "product_id": "P0001", "current_inventory": 0}
        res3 = self.client.post('/api/predict', json=payload_zero_stock)
        self.assertEqual(res3.status_code, 200)
        self.assertEqual(res3.get_json()["data"]["inventory"]["current_inventory"], 0)

    # =========================================================================
    # FEATURE 3: CSV VALIDATION & BATCH FORECASTING
    # =========================================================================
    def test_feature3_valid_csv_upload_and_preview(self):
        """Tests uploading a valid retail snapshot CSV and inspecting schema validation."""
        csv_content = (
            "Date,Store ID,Product ID,Category,Region,Inventory Level,Units Sold,Units Ordered,Demand Forecast,Price,Discount,Weather Condition,Holiday/Promotion,Competitor Pricing,Seasonality\n"
            "2023-10-01,S001,P0001,Groceries,North,200,120,50,115,35.0,5,Sunny,0,34.0,Autumn\n"
            "2023-10-02,S001,P0001,Groceries,North,180,130,50,125,35.0,5,Rainy,0,34.0,Autumn\n"
        )
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_upload.csv')}
        res = self.client.post('/api/upload_csv', content_type='multipart/form-data', data=data)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["success"])
        self.assertTrue(body["is_valid"], "Schema must be marked valid when required columns exist")
        self.assertEqual(body["row_count"], 2)
        self.assertEqual(body["total_nulls"], 0)
        self.assertEqual(len(body["preview"]), 2)

    def test_feature3_csv_validation_missing_columns(self):
        """Rejects or flags CSV when required columns are absent."""
        bad_csv = "Date,Store ID,Price\n2023-10-01,S001,35.0\n"
        data = {'file': (io.BytesIO(bad_csv.encode('utf-8')), 'bad_schema.csv')}
        res = self.client.post('/api/upload_csv', content_type='multipart/form-data', data=data)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertFalse(body["is_valid"], "Must be marked invalid if required columns are missing")
        self.assertIn("Product ID", body["missing_required_columns"])

    def test_feature3_non_csv_file_rejection(self):
        """Rejects non-CSV extensions (.pdf, .exe)."""
        data = {'file': (io.BytesIO(b"binary content"), 'malicious.exe')}
        res = self.client.post('/api/upload_csv', content_type='multipart/form-data', data=data)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file format", res.get_json()["error"])

    def test_feature3_batch_forecast_execution(self):
        """Executes batch forecasting and verifies predictions and metrics."""
        res = self.client.post('/api/batch_forecast', json={"model_choice": "champion", "max_rows": 20})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["success"])
        self.assertGreater(body["processed_records"], 0)
        self.assertGreater(body["processing_time_sec"], 0)
        self.assertGreater(len(body["results"]), 0)

        # Verify predictions correspond to realistic bounds
        for row in body["results"]:
            self.assertIn("Predicted Demand", row)
            self.assertGreaterEqual(row["Predicted Demand"], 0)
            self.assertIn("Reorder Point", row)

    # =========================================================================
    # FEATURE 4: FORECAST HISTORY PERSISTENCE
    # =========================================================================
    def test_feature4_history_crud_and_persistence(self):
        """Tests end-to-end saving, querying, filtering, and deletion of forecast runs."""
        # 1. Clear test DB to known baseline
        clear_history()

        # 2. Generate a forecast via API
        payload = {"store_id": "S003", "product_id": "P0015", "price": 60.0}
        pred_res = self.client.post('/api/predict', json=payload)
        self.assertEqual(pred_res.status_code, 200)
        run_id = pred_res.get_json()["data"]["run_id"]
        self.assertTrue(run_id.startswith("RUN-"))

        # 3. Retrieve history and confirm saved record
        hist_res = self.client.get('/api/forecast_history')
        self.assertEqual(hist_res.status_code, 200)
        hist_data = hist_res.get_json()["data"]
        self.assertEqual(hist_data["total"], 1)
        saved_run = hist_data["runs"][0]
        self.assertEqual(saved_run["run_id"], run_id)
        self.assertEqual(saved_run["store_id"], "S003")
        self.assertEqual(saved_run["product_id"], "P0015")

        # 4. Delete the run and verify deletion
        del_res = self.client.delete(f'/api/forecast_history/{run_id}')
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(del_res.get_json()["success"])

        # 5. Confirm history is now empty
        empty_res = self.client.get('/api/forecast_history')
        self.assertEqual(empty_res.get_json()["data"]["total"], 0)

    # =========================================================================
    # FEATURE 5: INVENTORY RECOMMENDATION & OPERATIONS RESEARCH
    # =========================================================================
    def test_feature5_safety_stock_formula(self):
        """Independently validates SS = Z * sigma_d * sqrt(L)."""
        z_95 = 1.6449
        sigma = 20.0
        lead_time = 4
        expected_ss = int(np.ceil(z_95 * sigma * np.sqrt(lead_time))) # 1.6449 * 20 * 2 = 65.796 -> 66
        actual_ss = self.optimizer.calculate_safety_stock(std_daily_demand=sigma, lead_time_days=lead_time, service_level=0.95)
        self.assertEqual(actual_ss, expected_ss)

    def test_feature5_reorder_point_formula(self):
        """Independently validates ROP = (d * L) + SS."""
        d = 75.0
        lead_time = 3
        ss = 30
        expected_rop = int(np.ceil(d * lead_time + ss)) # 225 + 30 = 255
        actual_rop = self.optimizer.calculate_reorder_point(predicted_daily_demand=d, safety_stock=ss, lead_time_days=lead_time)
        self.assertEqual(actual_rop, expected_rop)

    def test_feature5_order_quantity_never_negative(self):
        """Verifies order quantity is non-negative and properly offsets incoming orders."""
        # Case A: Massive stock -> order quantity must be 0
        diag_high = self.optimizer.evaluate_inventory_health(
            current_inventory=1000,
            predicted_daily_demand=20.0,
            incoming_orders=200,
            lead_time_days=3
        )
        self.assertEqual(diag_high["recommended_order_qty"], 0)
        self.assertEqual(diag_high["status"], "OVERSTOCKED")

        # Case B: Zero stock and zero incoming -> order quantity must be positive
        diag_zero = self.optimizer.evaluate_inventory_health(
            current_inventory=0,
            predicted_daily_demand=100.0,
            incoming_orders=0,
            lead_time_days=3
        )
        self.assertGreater(diag_zero["recommended_order_qty"], 0)
        self.assertEqual(diag_zero["status"], "CRITICAL_STOCKOUT")

    def test_feature5_store_inventory_matrix_api(self):
        """Tests the full 20-SKU store inventory matrix endpoint."""
        res = self.client.get('/api/inventory_matrix?store_id=S001&model_choice=champion')
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(len(body["matrix"]), 20, "Must return diagnostic for all 20 products in the store")

        for item in body["matrix"]:
            self.assertIn("safety_stock", item)
            self.assertIn("reorder_point", item)
            self.assertIn("recommended_order", item)
            self.assertGreaterEqual(item["recommended_order"], 0)

    # =========================================================================
    # FEATURE 6: DEMAND BY CATEGORY
    # =========================================================================
    def test_feature6_category_demand_aggregation(self):
        """Verifies market share percentages and category demand totals."""
        res = self.client.get('/api/analytics/category_demand')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]

        cats = data["categories"]
        expected_cats = ['Clothing', 'Electronics', 'Furniture', 'Groceries', 'Toys']
        self.assertEqual(sorted(cats), sorted(expected_cats))

        # Check market share sum ≈ 100%
        total_share = sum(data["market_share_pct"])
        self.assertAlmostEqual(total_share, 100.0, places=0)

        # Check totals sum to catalog volume
        self.assertEqual(sum(data["total_units"]), data["total_catalog_units"])

    # =========================================================================
    # FEATURE 7: SALES TREND & SEASONALITY ANALYSIS
    # =========================================================================
    def test_feature7_sales_trends_and_seasonality(self):
        """Verifies monthly, day-of-week, seasonal, and promotion aggregations."""
        res = self.client.get('/api/analytics/sales_trends?store_id=all')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]

        # Check 7 days of week
        self.assertEqual(len(data["day_of_week"]["labels"]), 7)
        self.assertEqual(len(data["day_of_week"]["average_sales"]), 7)

        # Check 4 seasons
        self.assertEqual(len(data["seasonality"]["labels"]), 4)

        # Check promotion comparison
        promo = data["promotion_comparison"]
        self.assertGreater(promo["regular_mean"], 0)
        self.assertGreater(promo["promotion_mean"], 0)

        # Check summary metrics
        self.assertGreater(data["summary"]["total_units_sold"], 0)
        self.assertGreater(data["summary"]["average_daily_demand"], 0)

    # =========================================================================
    # FEATURE 8: PRICE & PROMOTION IMPACT / WHAT-IF SIMULATOR
    # =========================================================================
    def test_feature8_pricing_analytics_and_correlations(self):
        """Verifies price scatter sample, discount tiers, and correlation coefficients."""
        res = self.client.get('/api/analytics/price_promotion')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]

        # Check scatter points length
        self.assertGreater(len(data["scatter_points"]), 50)
        for pt in data["scatter_points"]:
            self.assertIn("x", pt) # Price
            self.assertIn("y", pt) # Units Sold
            self.assertGreater(pt["x"], 0)

        # Check discount tiers
        self.assertIn("discount_impact", data)
        self.assertGreater(len(data["discount_impact"]["labels"]), 0)

    def test_feature8_what_if_simulation_response(self):
        """Tests live model inference reaction to price and discount adjustment."""
        # Baseline simulation
        res1 = self.client.post('/api/predict', json={"store_id": "S001", "product_id": "P0001", "price": 40.0, "discount": 0})
        # Scenario simulation with discount
        res2 = self.client.post('/api/predict', json={"store_id": "S001", "product_id": "P0001", "price": 40.0, "discount": 25})

        u1 = res1.get_json()["data"]["forecast"]["predicted_units_sold"]
        u2 = res2.get_json()["data"]["forecast"]["predicted_units_sold"]
        self.assertGreater(u1, 0)
        self.assertGreater(u2, 0)

    # =========================================================================
    # FEATURE 9: STOCKOUT RISK ALERTS
    # =========================================================================
    def test_feature9_stockout_risk_bounds_and_ranking(self):
        """Verifies stockout risk probabilities are in [0, 100] and properly ranked."""
        res = self.client.get('/api/analytics/stockout_alerts')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]

        self.assertEqual(data["total_items"], 100, "Must evaluate all 100 store SKUs")
        alerts = data["alerts"]

        for a in alerts:
            # Risk % must be bounded
            self.assertGreaterEqual(a["stockout_risk_pct"], 0.0)
            self.assertLessEqual(a["stockout_risk_pct"], 100.0)
            self.assertIn(a["severity"], ["danger", "warning", "info", "success"])
            self.assertGreater(len(a["reason"]), 0)

        # Severity filter test
        res_crit = self.client.get('/api/analytics/stockout_alerts?severity=danger')
        self.assertEqual(res_crit.status_code, 200)
        for a in res_crit.get_json()["data"]["alerts"]:
            self.assertEqual(a["severity"], "danger")

    # =========================================================================
    # FEATURE 10: MODEL BENCHMARKING & EXPLAINABLE AI (XAI)
    # =========================================================================
    def test_feature10_benchmark_metrics_accuracy(self):
        """Verifies the college evaluation matrix matches the test set results."""
        res = self.client.get('/api/benchmark_metrics')
        self.assertEqual(res.status_code, 200)
        metrics = res.get_json()["metrics"]

        # Check required models are present
        self.assertIn("Linear Regression (Baseline)", metrics)
        self.assertIn("LightGBM Regressor (Champion)", metrics)
        self.assertIn("Random Forest Regressor", metrics)
        self.assertIn("Legacy Model (Demand Forecast Target, Leaked Units Sold)", metrics)

        # Verify verified scores
        lr = metrics["Linear Regression (Baseline)"]
        self.assertAlmostEqual(lr["R2"], 0.3448, places=3)
        self.assertAlmostEqual(lr["MAE"], 68.5861, places=1)

        lgb = metrics["LightGBM Regressor (Champion)"]
        self.assertAlmostEqual(lgb["R2"], 0.3408, places=3)

    def test_feature10_explainable_ai_feature_importance(self):
        """Verifies extraction of LightGBM tree importances and LR coefficients."""
        res = self.client.get('/api/analytics/explainability')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]

        # Check LightGBM feature importance list
        lgb_imp = data["lightgbm_feature_importance"]
        self.assertGreater(len(lgb_imp), 0)
        for item in lgb_imp:
            self.assertIn("feature", item)
            self.assertIn("importance_score", item)
            self.assertGreater(item["importance_score"], 0)

        # Check LR coefficients
        lr_coef = data["linear_regression_top_coefficients"]
        self.assertGreater(len(lr_coef), 0)
        for c in lr_coef:
            self.assertIn("feature", c)
            self.assertIn("coefficient", c)

    # =========================================================================
    # FEATURE 11: REPORTS & EXPORT HUB
    # =========================================================================
    def test_feature11_all_pdf_reports(self):
        """Validates generation and byte integrity of all 5 PDF reports."""
        pdf_reports = [
            "forecast",
            "inventory",
            "benchmark",
            "category",
            "academic_summary"
        ]
        for r_name in pdf_reports:
            res = self.client.get(f'/api/export/pdf/{r_name}')
            self.assertEqual(res.status_code, 200, f"PDF report {r_name} failed")
            self.assertEqual(res.mimetype, 'application/pdf')
            self.assertTrue(res.data.startswith(b'%PDF-'), f"Report {r_name} missing valid PDF signature")
            self.assertGreater(len(res.data), 1500, f"Report {r_name} file size unexpectedly small")

    def test_feature11_invalid_export_types(self):
        """Validates that unsupported export types return HTTP 400."""
        res_csv = self.client.get('/api/export/csv/nonexistent_type')
        self.assertEqual(res_csv.status_code, 400)
        
        res_pdf = self.client.get('/api/export/pdf/nonexistent_type')
        self.assertEqual(res_pdf.status_code, 400)

    def test_feature2_model_fallback(self):
        """Fallback to champion model when model_choice is unknown."""
        payload = {"store_id": "S001", "product_id": "P0001", "model_choice": "unknown_model_name"}
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("LightGBM", res.get_json()["data"]["model_used"])

    def test_feature11_all_csv_exports(self):
        """Validates generation and data integrity of all 4 CSV exports."""
        csv_types = [
            "forecast_history",
            "stockout_alerts",
            "batch_forecast",
            "inventory_recommendations"
        ]
        for c_type in csv_types:
            res = self.client.get(f'/api/export/csv/{c_type}')
            self.assertEqual(res.status_code, 200, f"CSV export {c_type} failed")
            self.assertEqual(res.mimetype, 'text/csv')
            csv_str = res.data.decode('utf-8')
            self.assertGreater(len(csv_str), 0)

    def test_dataset_integrity(self):
        """Validates dataset record counts (73,100), schema (15 cols), and 0 nulls."""
        data_path = os.path.join("archive", "retail_store_inventory.csv")
        df = pd.read_csv(data_path)
        self.assertEqual(len(df), 73100)
        self.assertEqual(df.shape[1], 15)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_chronological_split(self):
        """Verifies exact 80/20 chronological time-aware split at 2023-08-08."""
        data_path = os.path.join("archive", "retail_store_inventory.csv")
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        unique_dates = df['Date'].sort_values().unique()
        split_idx = int(len(unique_dates) * 0.8)
        cutoff_date = unique_dates[split_idx]
        train_rows = (df['Date'] < cutoff_date).sum()
        test_rows = (df['Date'] >= cutoff_date).sum()
        self.assertEqual(train_rows, 58400)
        self.assertEqual(test_rows, 14700)
        self.assertEqual(pd.to_datetime(cutoff_date).strftime('%Y-%m-%d'), "2023-08-08")


    def test_indian_numbering_formatting(self):
        """Verifies exact Indian digit grouping requirements: 1K, 1 Lakh, 10 Lakhs, 1 Crore."""
        from currency_utils import format_inr
        self.assertEqual(format_inr(1000, 0), "₹1,000")
        self.assertEqual(format_inr(100000, 0), "₹1,00,000")
        self.assertEqual(format_inr(1000000, 0), "₹10,00,000")
        self.assertEqual(format_inr(10000000, 0), "₹1,00,00,000")
        
        # With decimals
        self.assertEqual(format_inr(1000, 2), "₹1,000.00")
        self.assertEqual(format_inr(100000, 2), "₹1,00,000.00")
        self.assertEqual(format_inr(6502.50, 2), "₹6,502.50")
        self.assertEqual(format_inr(111732000, 0), "₹11,17,32,000")

    def test_api_currency_vs_non_monetary(self):
        """Verifies currency is applied only to monetary values, never to counts/metrics."""
        payload = {"store_id": "S001", "product_id": "P0001", "price": 100000}
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]

        # Expected revenue is numeric internally
        self.assertIsInstance(data["forecast"]["expected_revenue"], (int, float))
        # Non-monetary units must be numeric ints/floats
        self.assertIsInstance(data["forecast"]["predicted_units_sold"], int)
        self.assertIsInstance(data["inventory"]["safety_stock"], int)
        self.assertIsInstance(data["inventory"]["reorder_point"], int)
        self.assertIsInstance(data["inventory"]["stockout_risk_pct"], (int, float))

        # Check feature attribution for Unit Price has ₹
        price_contrib = next(f for f in data["feature_contributions"] if f["feature"] == "Unit Price")
        self.assertTrue(price_contrib["value"].startswith("₹1,00,000"))

    def test_pdf_reports_inr_encoding(self):
        """Verifies all PDF generators render Indian Rupee (₹) without UnicodeEncodeError."""
        for r_type in ["forecast", "inventory", "benchmark", "category_demand", "academic_summary"]:
            res = self.client.get(f'/api/export/pdf/{r_type}')
            self.assertEqual(res.status_code, 200, f"PDF export {r_type} failed")
            self.assertEqual(res.mimetype, 'application/pdf')
            self.assertGreater(len(res.data), 1000)

    def test_csv_inr_headers_and_formatting(self):
        """Verifies CSV exports include INR ₹ headers and Indian grouped prices."""
        res_inv = self.client.get('/api/export/csv/inventory_recommendations?store_id=S001')
        self.assertEqual(res_inv.status_code, 200)
        csv_text = res_inv.data.decode('utf-8')
        self.assertIn("Unit Price (INR ₹)", csv_text)
        self.assertIn("Estimated Replenishment Cost (INR ₹)", csv_text)
        self.assertIn("₹", csv_text)

        res_batch = self.client.get('/api/export/csv/batch_forecast')
        self.assertEqual(res_batch.status_code, 200)

    # =========================================================================
    # FEATURE 12: ENTERPRISE AUTHENTICATION & LOGIN PORTAL
    # =========================================================================
    def test_feature12_login_page_renders_successfully(self):
        """Verifies that the dedicated /login portal loads with full branding & forms."""
        res = self.client.get('/login')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("StockSense", html)
        self.assertIn("AI", html)
        self.assertIn("Make smarter", html)
        self.assertIn("inventory decisions", html)
        self.assertIn("Welcome back", html)
        self.assertIn("demo@stocksense.ai", html)
        self.assertIn("signinPassword", html)
        self.assertIn("signinEmail", html)
        self.assertIn("btnSignInSubmit", html)

    def test_feature12_unauthenticated_dashboard_redirect(self):
        """Verifies unauthenticated requests to dashboard redirect to /login."""
        # Using a fresh test client with testing disabled to verify authentication guard
        unauth_client = app.app.test_client()
        orig_testing = app.app.testing
        try:
            app.app.testing = False
            res = unauth_client.get('/')
            self.assertEqual(res.status_code, 302)
            self.assertIn('/login', res.headers.get('Location', ''))
        finally:
            app.app.testing = orig_testing

    def test_feature12_invalid_credentials_rejected(self):
        """Verifies invalid email or password returns 401 with informative error."""
        res = self.client.post('/api/auth/login', json={
            "email": "demo@stocksense.ai",
            "password": "IncorrectPassword123"
        })
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Invalid email or password", data["error"])

    def test_feature12_missing_credentials_rejected(self):
        """Verifies missing email or password returns 400."""
        res = self.client.post('/api/auth/login', json={"email": ""})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])

    def test_feature12_successful_demo_login_and_session(self):
        """Verifies successful login establishes session and enables access."""
        auth_client = app.app.test_client()
        res = auth_client.post('/api/auth/login', json={
            "email": "demo@stocksense.ai",
            "password": "StockSense2026!",
            "remember_me": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["user"]["email"], "demo@stocksense.ai")
        self.assertEqual(data["user"]["full_name"], "Jaswin")

        # Verify /api/auth/me returns authenticated state
        me_res = auth_client.get('/api/auth/me')
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.get_json()
        self.assertTrue(me_data["authenticated"])
        self.assertEqual(me_data["user"]["full_name"], "Jaswin")

        # Verify authenticated access to / returns 200
        orig_testing = app.app.testing
        try:
            app.app.testing = False
            dash_res = auth_client.get('/')
            self.assertEqual(dash_res.status_code, 200)

            # Accessing /login while authenticated redirects to /
            login_res = auth_client.get('/login')
            self.assertEqual(login_res.status_code, 302)
            self.assertEqual(login_res.headers.get('Location', ''), '/')
        finally:
            app.app.testing = orig_testing

    def test_feature12_user_logout(self):
        """Verifies logout terminates active session."""
        auth_client = app.app.test_client()
        # Login first
        auth_client.post('/api/auth/login', json={
            "email": "admin@stocksense.ai",
            "password": "AdminStock#2026"
        })
        # Logout via API
        logout_res = auth_client.post('/api/auth/logout')
        self.assertEqual(logout_res.status_code, 200)
        self.assertTrue(logout_res.get_json()["success"])

        # Check /api/auth/me is now unauthenticated
        me_res = auth_client.get('/api/auth/me')
        self.assertFalse(me_res.get_json()["authenticated"])

        # Logout via GET /logout page redirects to /login?logout=1
        page_logout = auth_client.get('/logout')
        self.assertEqual(page_logout.status_code, 302)
        self.assertIn('logout=1', page_logout.headers.get('Location', ''))

    def test_feature12_user_registration_and_duplicate_check(self):
        """Verifies new account registration and uniqueness constraint."""
        import time
        unique_email = f"analyst_{int(time.time()*1000)}@enterprise-test.com"
        res = self.client.post('/api/auth/register', json={
            "full_name": "Marcus Wright",
            "email": unique_email,
            "role": "Supply Chain Planner",
            "password": "SecurePassword#2026"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["user"]["email"], unique_email)

        # Duplicate registration with same email should fail with 400
        dup_res = self.client.post('/api/auth/register', json={
            "full_name": "Marcus Wright Duplicate",
            "email": unique_email,
            "role": "Supply Chain Planner",
            "password": "AnotherPassword#2026"
        })
        self.assertEqual(dup_res.status_code, 400)
        self.assertFalse(dup_res.get_json()["success"])

    def test_feature12_password_recovery_and_reset(self):
        """Verifies token-based password recovery and reset workflow."""
        # Request recovery token
        req_res = self.client.post('/api/auth/forgot-password', json={
            "email": "demo@stocksense.ai"
        })
        self.assertEqual(req_res.status_code, 200)
        token = req_res.get_json().get("token")
        self.assertIsNotNone(token)

        # Apply password reset
        reset_res = self.client.post('/api/auth/reset-password', json={
            "token": token,
            "new_password": "NewUpdatedPassword#2026"
        })
        self.assertEqual(reset_res.status_code, 200)
        self.assertTrue(reset_res.get_json()["success"])

        # Verify old password fails
        old_login = self.client.post('/api/auth/login', json={
            "email": "demo@stocksense.ai",
            "password": "StockSense2026!"
        })
        self.assertEqual(old_login.status_code, 401)

        # Verify new password succeeds
        new_login = self.client.post('/api/auth/login', json={
            "email": "demo@stocksense.ai",
            "password": "NewUpdatedPassword#2026"
        })
        self.assertEqual(new_login.status_code, 200)

        # Reset back to default demo password for other tests and users
        import auth_db
        from werkzeug.security import generate_password_hash
        conn = auth_db.get_connection()
        with conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (generate_password_hash("StockSense2026!"), "demo@stocksense.ai")
            )
        conn.close()

    # =========================================================================
    # FEATURE 12: GROUNDED EXPLAINABLE AI (TreeSHAP & Coefficient Attribution)
    # =========================================================================
    def test_feature12_treeshap_and_linear_local_attributions(self):
        """Verifies native TreeSHAP local attribution for LightGBM and standardized effects for LR."""
        # 1. Test LightGBM TreeSHAP
        res_lgb = self.engine.predict_demand(
            store_id="S001",
            product_id="P0001",
            model_choice="champion"
        )
        attribs_lgb = res_lgb.get("feature_contributions", [])
        self.assertGreaterEqual(len(attribs_lgb), 5, "Must return at least 5 local attribution components")
        
        feature_names = [a["feature"] for a in attribs_lgb]
        self.assertIn("Unit Price", feature_names)
        self.assertIn("Current Inventory", feature_names)
        self.assertIn("Base Demand", feature_names)

        for a in attribs_lgb:
            self.assertIsInstance(a["impact_units"], (int, float), "Attribution impact must be a numerical scalar")

        # 2. Test Linear Regression Standardized Effects
        res_lr = self.engine.predict_demand(
            store_id="S001",
            product_id="P0001",
            model_choice="baseline_lr"
        )
        attribs_lr = res_lr.get("feature_contributions", [])
        self.assertGreaterEqual(len(attribs_lr), 5)
        for a in attribs_lr:
            self.assertIsInstance(a["impact_units"], (int, float))

    # =========================================================================
    # FEATURE 13: STOCHASTIC LEAD TIME OPERATIONS RESEARCH BUFFER
    # =========================================================================
    def test_feature13_stochastic_lead_time_safety_stock(self):
        """Verifies that stochastic lead time variance increases safety buffer appropriately."""
        # Deterministic lead time (sigma_L = 0)
        ss_det = self.optimizer.calculate_safety_stock(
            std_daily_demand=25.0,
            lead_time_days=3,
            service_level=0.95,
            lead_time_std_days=0.0
        )

        # Stochastic lead time (sigma_L = 1.0 day supplier delay risk)
        ss_stoch = self.optimizer.calculate_safety_stock(
            std_daily_demand=25.0,
            lead_time_days=3,
            service_level=0.95,
            lead_time_std_days=1.0,
            mean_daily_demand=100.0
        )

        self.assertGreater(
            ss_stoch,
            ss_det,
            f"Stochastic lead time SS ({ss_stoch}) must be strictly greater than deterministic SS ({ss_det})"
        )

        # Verify API level integration
        health_eval = self.optimizer.evaluate_inventory_health(
            current_inventory=200,
            predicted_daily_demand=60.0,
            std_daily_demand=25.0,
            lead_time_days=3,
            lead_time_std_days=0.8
        )
        self.assertEqual(health_eval["lead_time_std_days"], 0.8)
        self.assertGreater(health_eval["safety_stock"], 0)


if __name__ == '__main__':
    unittest.main()


