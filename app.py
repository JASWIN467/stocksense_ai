"""
StockSense AI - Flask Web Application Backend
Production-grade REST server connecting ML pipelines, Operations Research policies,
Advanced Analytics, Persistent History, CSV Batch Validation, and PDF Report Exports.
"""

import os
import io
import json
import time
import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response, session, redirect, url_for

from inference import StockSenseInference, get_engine
from inventory_engine import InventoryOptimizer
from analytics_engine import get_analytics
from history_db import init_db, save_forecast_run, get_history_runs, delete_run, clear_history
import auth_db
import pdf_generator
from currency_utils import format_inr

app = Flask(__name__)

# Security & Session Configuration
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "stocksense-production-enterprise-secret-key-2026")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_NAME="stocksense_session",
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=30)
)

engine = get_engine()
analytics = get_analytics()

# Ensure history and auth DBs exist
init_db()
auth_db.init_auth_db()

# Pre-load historical dataset for time-series lookups
data_path = os.path.join("archive", "retail_store_inventory.csv")
history_df = None
if os.path.exists(data_path):
    try:
        history_df = pd.read_csv(data_path)
        history_df['Date'] = pd.to_datetime(history_df['Date'])
    except Exception as e:
        print("Warning: Could not pre-load historical CSV:", e)

# In-memory store for recent uploaded batch predictions & uploaded dataframe
latest_batch_results = []
uploaded_batch_df = None


def is_authenticated():
    """Checks whether the client session has a verified user ID or testing mode is active."""
    if app.testing or app.config.get("TESTING", False):
        return True
    return "user_id" in session


# =============================================================================
# AUTHENTICATION & PORTAL ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serves the main application dashboard, protected by authentication session."""
    if not is_authenticated():
        return redirect("/login")
    return render_template("index.html")


@app.route("/login")
def login():
    """Serves the dedicated StockSense AI enterprise login portal."""
    if is_authenticated() and not app.testing:
        next_url = request.args.get("next") or "/"
        return redirect(next_url)
    return render_template("login.html")


@app.route("/logout")
def logout_page():
    """Logs out the active session and redirects to the login portal."""
    session.clear()
    return redirect("/login?logout=1")


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Validates user credentials against stored password hashes in SQLite."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    remember_me = bool(data.get("remember_me", False))

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    user = auth_db.verify_user(email, password)
    if not user:
        return jsonify({"success": False, "error": "Invalid email or password. Please verify your credentials."}), 401

    # Establish secure session
    session.clear()
    session.permanent = remember_me
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user["full_name"]
    session["user_role"] = user["role"]

    return jsonify({
        "success": True,
        "message": "Authentication successful.",
        "user": user,
        "redirect": "/"
    })


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """Terminates user session."""
    session.clear()
    return jsonify({"success": True, "message": "Session terminated successfully."})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    """Returns profile of currently authenticated session user."""
    if not is_authenticated() or "user_id" not in session:
        return jsonify({"authenticated": False, "user": None})

    user = auth_db.get_user_by_id(session["user_id"])
    return jsonify({"authenticated": True, "user": user})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    """Registers a new enterprise user account."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    role = (data.get("role") or "Inventory Analyst").strip()

    success, msg, user = auth_db.create_user(email, password, full_name, role)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    # Auto-login newly registered user
    session.clear()
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user["full_name"]
    session["user_role"] = user["role"]

    return jsonify({
        "success": True,
        "message": msg,
        "user": user,
        "redirect": "/"
    })


@app.route("/api/auth/forgot-password", methods=["POST"])
def auth_forgot_password():
    """Initiates secure password recovery workflow."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip()
    success, msg, token = auth_db.create_password_reset_request(email)
    return jsonify({"success": success, "message": msg, "token": token})


@app.route("/api/auth/reset-password", methods=["POST"])
def auth_reset_password():
    """Updates password using verified reset token."""
    data = request.get_json() or {}
    token = (data.get("token") or "").strip()
    new_password = data.get("new_password") or ""

    success, msg = auth_db.reset_password_with_token(token, new_password)
    if not success:
        return jsonify({"success": False, "error": msg}), 400
    return jsonify({"success": True, "message": msg})


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Returns store and product metadata for frontend dropdowns."""
    return jsonify(engine.metadata)



@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predicts demand and evaluates inventory health for a specific configuration.
    Automatically persists real execution to SQLite forecast history.
    """
    data = request.get_json() or {}
    try:
        res = engine.predict_demand(
            store_id=str(data.get("store_id", "S001")),
            product_id=str(data.get("product_id", "P0001")),
            date=data.get("date"),
            price=float(data.get("price")) if data.get("price") is not None else None,
            discount=max(0, min(100, int(data.get("discount", 0)))),
            competitor_pricing=float(data.get("competitor_pricing")) if data.get("competitor_pricing") is not None else None,
            weather_condition=str(data.get("weather_condition", "Sunny")),
            holiday_promotion=int(data.get("holiday_promotion", 0)),
            seasonality=data.get("seasonality"),
            current_inventory=max(0, int(data.get("current_inventory"))) if data.get("current_inventory") is not None else None,
            incoming_orders=max(0, int(data.get("incoming_orders", 0))),
            lead_time_days=max(1, int(data.get("lead_time_days", 3))),
            lead_time_std_days=max(0.0, float(data.get("lead_time_std_days", 0.0))),
            service_level=max(0.50, min(0.999, float(data.get("service_level", 0.95)))),
            forecast_horizon_days=max(1, int(data.get("forecast_horizon_days", 7))),
            model_choice=str(data.get("model_choice", "champion"))
        )
        
        # Save run to persistent SQLite forecast history
        run_id = save_forecast_run(res)
        res["run_id"] = run_id

        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/inventory_matrix", methods=["GET"])
def inventory_matrix():
    """Returns full inventory health matrix across all 20 products for a store."""
    store_id = request.args.get("store_id", "S001")
    lead_time = max(1, int(request.args.get("lead_time_days", 3)))
    service_level = max(0.50, min(0.999, float(request.args.get("service_level", 0.95))))
    horizon = max(1, int(request.args.get("forecast_horizon_days", 7)))
    model_choice = request.args.get("model_choice", "champion")

    snapshot = engine.snapshot_df
    if snapshot.empty:
        return jsonify({"success": False, "error": "No snapshot data available"}), 404

    store_rows = snapshot[snapshot['Store ID'] == store_id]
    matrix = []

    for _, row in store_rows.iterrows():
        p_id = row['Product ID']
        curr_stock = int(row['Inventory Level'])
        price = float(row['Price'])

        pred_res = engine.predict_demand(
            store_id=store_id,
            product_id=p_id,
            price=price,
            current_inventory=curr_stock,
            lead_time_days=lead_time,
            service_level=service_level,
            forecast_horizon_days=horizon,
            model_choice=model_choice
        )

        inv = pred_res["inventory"]
        matrix.append({
            "product_id": p_id,
            "product_name": pred_res["product_name"],
            "category": pred_res["category"],
            "price": price,
            "current_inventory": curr_stock,
            "incoming_orders": inv.get("incoming_orders", 0),
            "inventory_position": inv.get("inventory_position", curr_stock),
            "predicted_demand": pred_res["forecast"]["predicted_units_sold"],
            "safety_stock": inv["safety_stock"],
            "reorder_point": inv["reorder_point"],
            "order_up_to_level": inv["order_up_to_level"],
            "economic_order_qty": inv["economic_order_qty"],
            "stockout_risk_pct": inv["stockout_risk_pct"],
            "recommended_order": inv["recommended_order_qty"],
            "status": inv["status"],
            "status_label": inv["status_label"],
            "severity": inv["severity"],
            "action": inv["action_recommendation"]
        })

    return jsonify({
        "success": True,
        "store_id": store_id,
        "total_skus": len(matrix),
        "lead_time_days": lead_time,
        "service_level": service_level,
        "matrix": matrix
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns past 30 days actual sales + 7 days forward forecast projection."""
    global history_df
    store_id = request.args.get("store_id", "S001")
    product_id = request.args.get("product_id", "P0001")
    model_choice = request.args.get("model_choice", "champion")

    if history_df is None or history_df.empty:
        return jsonify({"success": False, "error": "Historical data unavailable"}), 404

    sku_history = history_df[
        (history_df['Store ID'] == store_id) & 
        (history_df['Product ID'] == product_id)
    ].sort_values('Date').tail(30)

    if sku_history.empty:
        return jsonify({"success": False, "error": f"No historical data found for Store {store_id} and Product {product_id}"}), 404

    dates = [d.strftime('%Y-%m-%d') for d in sku_history['Date']]
    actual_sales = sku_history['Units Sold'].tolist()
    inventory_levels = sku_history['Inventory Level'].tolist()

    # Generate 7-day future projection
    last_date = sku_history['Date'].max()
    forecast_dates = []
    forecast_values = []
    ci_lower_list = []
    ci_upper_list = []

    for i in range(1, 8):
        f_date = (last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        forecast_dates.append(f_date)
        res = engine.predict_demand(
            store_id=store_id,
            product_id=product_id,
            date=f_date,
            model_choice=model_choice
        )
        val = res["forecast"]["predicted_units_sold"]
        forecast_values.append(val)
        ci_lower_list.append(res["forecast"]["confidence_interval_95"]["lower"])
        ci_upper_list.append(res["forecast"]["confidence_interval_95"]["upper"])

    mean_30d = round(float(np.mean(actual_sales)), 1) if actual_sales else 0.0
    mean_forecast = round(float(np.mean(forecast_values)), 1) if forecast_values else 0.0
    growth_pct = round(((mean_forecast - mean_30d) / max(1.0, mean_30d)) * 100.0, 1)

    return jsonify({
        "success": True,
        "store_id": store_id,
        "product_id": product_id,
        "stats": {
            "mean_30d": mean_30d,
            "mean_forecast": mean_forecast,
            "growth_pct": growth_pct
        },
        "historical": {
            "dates": dates,
            "units_sold": actual_sales,
            "inventory_level": inventory_levels
        },
        "forecast": {
            "dates": forecast_dates,
            "units_sold": forecast_values,
            "ci_lower": ci_lower_list,
            "ci_upper": ci_upper_list
        }
    })


@app.route("/api/benchmark_metrics", methods=["GET"])
def benchmark_metrics():
    """Returns college model comparison matrix."""
    metrics_path = os.path.join("models", "baseline_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"success": True, "metrics": data})
    return jsonify({"success": False, "error": "Metrics file not found"}), 404


@app.route("/api/forecast_history", methods=["GET"])
def get_forecast_history():
    """Returns persistent list of forecast execution runs."""
    store_id = request.args.get("store_id")
    product_id = request.args.get("product_id")
    search = request.args.get("search")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))

    data = get_history_runs(store_id=store_id, product_id=product_id, search=search, limit=limit, offset=offset)
    return jsonify({"success": True, "data": data})


@app.route("/api/forecast_history/<run_id>", methods=["DELETE"])
def delete_forecast_history_run(run_id):
    """Deletes a specific forecast history record."""
    success = delete_run(run_id)
    return jsonify({"success": success})


@app.route("/api/forecast_history/clear", methods=["POST"])
def clear_forecast_history():
    """Clears all historical forecast records."""
    count = clear_history()
    return jsonify({"success": True, "deleted_count": count})


@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    """
    Validates an uploaded retail inventory CSV dataset.
    Checks file extension, headers, missing values, duplicates, and returns preview rows.
    """
    global uploaded_batch_df
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded. Please select a CSV file."}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({"success": False, "error": "Unsupported file format. Please upload a CSV file (.csv)"}), 400

    try:
        # Seek to beginning in case file stream was read or buffered previously
        file.seek(0)
        raw_bytes = file.read()
        if not raw_bytes or len(raw_bytes.strip()) == 0:
            return jsonify({
                "success": False,
                "error": "The uploaded CSV file is empty (0 bytes). Please ensure the file contains valid CSV data with column headers."
            }), 400

        # Attempt decoding across common encodings
        decoded_text = None
        for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                decoded_text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            return jsonify({
                "success": False,
                "error": "Could not decode CSV text. Please ensure the file is saved in UTF-8 or standard text format."
            }), 400

        # Filter empty lines
        non_empty_lines = [line for line in decoded_text.splitlines() if line.strip()]
        if not non_empty_lines:
            return jsonify({
                "success": False,
                "error": "The uploaded CSV file contains no readable text or data rows."
            }), 400

        # Detect separator if commas are absent in the first non-empty line
        first_line = non_empty_lines[0]
        sep = ','
        if ',' not in first_line:
            if ';' in first_line:
                sep = ';'
            elif '\t' in first_line:
                sep = '\t'
            elif '|' in first_line:
                sep = '|'

        try:
            df_uploaded = pd.read_csv(io.StringIO(decoded_text), sep=sep, skip_blank_lines=True)
            # If parsed as a single column and semicolons or tabs exist, re-parse
            if len(df_uploaded.columns) <= 1 and (';' in first_line or '\t' in first_line):
                alt_sep = ';' if ';' in first_line else '\t'
                df_uploaded = pd.read_csv(io.StringIO(decoded_text), sep=alt_sep, skip_blank_lines=True)
        except pd.errors.EmptyDataError:
            return jsonify({
                "success": False,
                "error": "Failed to parse CSV: The file does not contain any column headers or data rows. Please ensure your CSV has a valid header line."
            }), 400
        except Exception:
            try:
                df_uploaded = pd.read_csv(io.StringIO(decoded_text), sep=None, engine='python', skip_blank_lines=True)
            except Exception as pe:
                return jsonify({
                    "success": False,
                    "error": f"Failed to parse CSV: {str(pe)}. Please verify that the file is a properly formatted CSV."
                }), 400

        if df_uploaded is None or df_uploaded.empty:
            return jsonify({"success": False, "error": "Uploaded CSV file contains no data rows."}), 400

        # Clean column headers by stripping outer whitespace
        df_uploaded.columns = [str(c).strip() for c in df_uploaded.columns]

        row_count, col_count = df_uploaded.shape

        required_cols = ['Date', 'Store ID', 'Product ID', 'Category', 'Inventory Level', 'Price']
        present_cols = df_uploaded.columns.tolist()
        missing_required = [c for c in required_cols if c not in present_cols]

        null_counts = int(df_uploaded.isnull().sum().sum())
        duplicate_count = int(df_uploaded.duplicated().sum())

        is_valid = len(missing_required) == 0

        if is_valid:
            try:
                df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'])
            except Exception:
                pass
            uploaded_batch_df = df_uploaded
        else:
            uploaded_batch_df = None

        # Preview first 8 rows
        preview_rows = df_uploaded.head(8).fillna("").to_dict(orient="records")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "row_count": row_count,
            "col_count": col_count,
            "columns": present_cols,
            "missing_required_columns": missing_required,
            "total_nulls": null_counts,
            "duplicate_rows": duplicate_count,
            "is_valid": is_valid,
            "preview": preview_rows
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to parse CSV: {str(e)}"}), 400


@app.route("/api/download_sample_csv", methods=["GET"])
def download_sample_csv():
    """Serves a valid sample retail inventory CSV for easy testing and template download."""
    sample_path = os.path.join("data", "sample_inventory_snapshot.csv")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("archive", "retail_store_inventory.csv")

    if os.path.exists(sample_path):
        return send_file(
            sample_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="stocksense_sample_inventory_template.csv"
        )
    return jsonify({"error": "Sample template CSV not found on server."}), 404


@app.route("/api/batch_forecast", methods=["POST"])
def batch_forecast():
    """
    Runs batch predictions on the uploaded CSV data or default test batch.
    """
    global latest_batch_results, uploaded_batch_df, history_df
    start_time = time.time()
    data = request.get_json() or {}
    model_choice = data.get("model_choice", "champion")
    
    # Process up to 500 rows for instantaneous response, or full batch
    max_rows = int(data.get("max_rows", 200))

    if uploaded_batch_df is not None and not uploaded_batch_df.empty:
        # Check if required columns exist in uploaded CSV
        req = ['Date', 'Store ID', 'Product ID', 'Price', 'Inventory Level']
        if not all(c in uploaded_batch_df.columns for c in req):
            return jsonify({"success": False, "error": "Uploaded CSV lacks required prediction columns."}), 400
        test_slice = uploaded_batch_df.head(max_rows).copy()
    elif history_df is not None and not history_df.empty:
        test_slice = history_df.tail(max_rows).copy()
    else:
        return jsonify({"success": False, "error": "No dataset available for batch forecasting."}), 404
    results = []

    for _, row in test_slice.iterrows():
        s_id = row['Store ID']
        p_id = row['Product ID']
        curr_stock = int(row['Inventory Level'])
        price = float(row['Price'])
        disc = int(row.get('Discount', 0))

        pred = engine.predict_demand(
            store_id=s_id,
            product_id=p_id,
            date=row['Date'].strftime('%Y-%m-%d'),
            price=price,
            discount=disc,
            current_inventory=curr_stock,
            model_choice=model_choice
        )
        actual = int(row['Units Sold']) if 'Units Sold' in row else 0
        predicted = pred["forecast"]["predicted_units_sold"]
        abs_err = abs(actual - predicted)
        
        results.append({
            "Date": row['Date'].strftime('%Y-%m-%d'),
            "Store ID": s_id,
            "Product ID": p_id,
            "Category": row.get('Category', 'Groceries'),
            "Inventory Level": curr_stock,
            "Price (INR ₹)": format_inr(price, 2),
            "Discount": disc,
            "Actual Units Sold": actual,
            "Predicted Demand": predicted,
            "Safety Stock": pred["inventory"]["safety_stock"],
            "Reorder Point": pred["inventory"]["reorder_point"],
            "Stockout Risk": f"{pred['inventory']['stockout_risk_pct']}%",
            "Recommended Order": pred["inventory"]["recommended_order_qty"],
            "Absolute Error": abs_err
        })

    latest_batch_results = results
    elapsed = round(time.time() - start_time, 2)
    mean_err = round(float(np.mean([r["Absolute Error"] for r in results])), 2)

    return jsonify({
        "success": True,
        "processed_records": len(results),
        "processing_time_sec": elapsed,
        "mean_absolute_error": mean_err,
        "model_used": model_choice,
        "results": results[:50] # return first 50 for fast rendering
    })


@app.route("/api/analytics/sales_trends", methods=["GET"])
def sales_trends():
    """Feature 7: Sales Trend & Seasonality Analytics."""
    store_id = request.args.get("store_id")
    category = request.args.get("category")
    res = analytics.get_sales_trends(store_id=store_id, category=category)
    return jsonify({"success": True, "data": res})


@app.route("/api/analytics/price_promotion", methods=["GET"])
def price_promotion():
    """Feature 8: Price & Promotion Impact Analytics."""
    category = request.args.get("category")
    res = analytics.get_price_promotion_impact(category=category)
    return jsonify({"success": True, "data": res})


@app.route("/api/analytics/category_demand", methods=["GET"])
def category_demand():
    """Feature: Category Demand & Market Share Analytics."""
    res = analytics.get_category_demand_breakdown()
    return jsonify({"success": True, "data": res})


@app.route("/api/analytics/stockout_alerts", methods=["GET"])
def stockout_alerts():
    """Feature 9: Multi-SKU Stockout Risk Monitoring & Ranking."""
    store_id = request.args.get("store_id")
    severity = request.args.get("severity")
    res = analytics.get_stockout_risk_alerts(store_id=store_id, severity_filter=severity)
    return jsonify({"success": True, "data": res})


@app.route("/api/analytics/explainability", methods=["GET"])
def explainability():
    """Feature 11: Explainable AI & Feature Importance."""
    res = analytics.get_model_explainability()
    return jsonify({"success": True, "data": res})


@app.route("/api/export/csv/<export_type>", methods=["GET"])
def export_csv(export_type):
    """Feature 10: High-fidelity CSV export for various application views."""
    output = io.StringIO()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if export_type == "forecast_history":
        runs = get_history_runs(limit=1000)["runs"]
        df_exp = pd.DataFrame(runs)
        if not df_exp.empty:
            if "unit_price" in df_exp.columns:
                df_exp["Unit Price (INR ₹)"] = df_exp["unit_price"].apply(lambda x: format_inr(x, 2) if pd.notnull(x) else "")
                df_exp.drop(columns=["unit_price"], inplace=True)
            if "expected_revenue" in df_exp.columns:
                df_exp["Expected Revenue (INR ₹)"] = df_exp["expected_revenue"].apply(lambda x: format_inr(x, 2) if pd.notnull(x) else "")
                df_exp.drop(columns=["expected_revenue"], inplace=True)
        filename = f"stocksense_forecast_history_{timestamp}.csv"
    elif export_type == "stockout_alerts":
        alerts = analytics.get_stockout_risk_alerts()["alerts"]
        df_exp = pd.DataFrame(alerts)
        if not df_exp.empty and "price" in df_exp.columns:
            df_exp["Price (INR ₹)"] = df_exp["price"].apply(lambda x: format_inr(x, 2) if pd.notnull(x) else "")
            df_exp.drop(columns=["price"], inplace=True)
        filename = f"stocksense_stockout_alerts_{timestamp}.csv"
    elif export_type == "batch_forecast":
        global latest_batch_results
        df_exp = pd.DataFrame(latest_batch_results if latest_batch_results else [])
        filename = f"stocksense_batch_forecast_{timestamp}.csv"
    elif export_type == "inventory_recommendations":
        store_id = request.args.get("store_id", "S001")
        # generate store matrix
        snap = engine.snapshot_df
        rows = snap[snap['Store ID'] == store_id]
        matrix = []
        for _, r in rows.iterrows():
            p_res = engine.predict_demand(store_id=store_id, product_id=r['Product ID'], current_inventory=int(r['Inventory Level']))
            unit_price = float(r.get('Price', 45.0))
            rec_order = p_res['inventory']['recommended_order_qty']
            est_order_cost = rec_order * unit_price
            matrix.append({
                "Store ID": store_id,
                "Product ID": r['Product ID'],
                "Category": p_res['category'],
                "Unit Price (INR ₹)": format_inr(unit_price, 2),
                "Current Stock": int(r['Inventory Level']),
                "Predicted Demand": p_res['forecast']['predicted_units_sold'],
                "Safety Stock": p_res['inventory']['safety_stock'],
                "Reorder Point": p_res['inventory']['reorder_point'],
                "Stockout Risk": f"{p_res['inventory']['stockout_risk_pct']}%",
                "Recommended Order": rec_order,
                "Estimated Replenishment Cost (INR ₹)": format_inr(est_order_cost, 2),
                "Status": p_res['inventory']['status_label']
            })
        df_exp = pd.DataFrame(matrix)
        filename = f"stocksense_inventory_matrix_{store_id}_{timestamp}.csv"
    else:
        return jsonify({"error": "Unknown export type"}), 400

    df_exp.to_csv(output, index=False)
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route("/api/export/pdf/<report_type>", methods=["GET"])
def export_pdf(report_type):
    """Feature 10: Publication-quality PDF Report Downloads."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        if report_type == "forecast":
            store_id = request.args.get("store_id", "S001")
            product_id = request.args.get("product_id", "P0001")
            model_choice = request.args.get("model_choice", "champion")
            pred_data = engine.predict_demand(store_id=store_id, product_id=product_id, model_choice=model_choice)
            pdf_buf = pdf_generator.generate_forecast_pdf(pred_data)
            filename = f"stocksense_forecast_{store_id}_{product_id}_{timestamp}.pdf"

        elif report_type == "inventory":
            store_id = request.args.get("store_id", "S001")
            # Generate store matrix items
            snap = engine.snapshot_df
            rows = snap[snap['Store ID'] == store_id]
            matrix_data = []
            for _, r in rows.iterrows():
                p_res = engine.predict_demand(store_id=store_id, product_id=r['Product ID'], current_inventory=int(r['Inventory Level']))
                matrix_data.append({
                    "product_id": r['Product ID'],
                    "category": p_res['category'],
                    "current_inventory": int(r['Inventory Level']),
                    "predicted_demand": p_res['forecast']['predicted_units_sold'],
                    "safety_stock": p_res['inventory']['safety_stock'],
                    "reorder_point": p_res['inventory']['reorder_point'],
                    "stockout_risk_pct": p_res['inventory']['stockout_risk_pct'],
                    "recommended_order": p_res['inventory']['recommended_order_qty'],
                    "status_label": p_res['inventory']['status_label']
                })
            pdf_buf = pdf_generator.generate_inventory_recommendation_pdf(store_id, matrix_data)
            filename = f"stocksense_inventory_recommendations_{store_id}_{timestamp}.pdf"

        elif report_type == "benchmark":
            metrics_path = os.path.join("models", "baseline_metrics.json")
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            pdf_buf = pdf_generator.generate_model_benchmark_pdf(metrics)
            filename = f"stocksense_academic_benchmark_audit_{timestamp}.pdf"

        elif report_type in ["category", "category_demand"]:
            cat_data = analytics.get_category_demand_breakdown()
            pdf_buf = pdf_generator.generate_category_analytics_pdf(cat_data)
            filename = f"stocksense_category_demand_{timestamp}.pdf"

        elif report_type == "academic_summary":
            metrics_path = os.path.join("models", "baseline_metrics.json")
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            cat_data = analytics.get_category_demand_breakdown()
            pdf_buf = pdf_generator.generate_academic_summary_pdf(metrics, cat_data)
            filename = f"stocksense_academic_summary_capstone_{timestamp}.pdf"

        else:
            return jsonify({"error": "Unknown report type"}), 400

        return send_file(
            pdf_buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


if __name__ == "__main__":
    import sys
    is_prod = "--prod" in sys.argv or os.environ.get("FLASK_ENV") == "production" or os.environ.get("ENVIRONMENT") == "production"

    print("==================================================================")
    print(f"  StockSense AI Server ({'Production WSGI Waitress' if is_prod else 'Standard Server'}) Running ...")
    print("  Local URL:  http://127.0.0.1:5000")
    print("  All Interfaces: http://0.0.0.0:5000")
    print("==================================================================")

    if is_prod:
        try:
            from waitress import serve
            print("  [Production WSGI] Serving with Waitress (6 worker threads)...")
            serve(app, host="0.0.0.0", port=5000, threads=6)
        except ImportError:
            print("  [Notice] Waitress not installed, launching standard server.")
            app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
