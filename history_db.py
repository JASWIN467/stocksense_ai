"""
StockSense AI - Forecast History Database Module
Persistent SQLite storage for authentic forecast runs, audits, and tracking.
"""

import sqlite3
import os
import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join("data", "forecast_history.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_history (
                run_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                store_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                category TEXT,
                target_date TEXT,
                forecast_horizon_days INTEGER,
                model_used TEXT,
                predicted_units INTEGER,
                current_inventory INTEGER,
                incoming_orders INTEGER,
                inventory_position INTEGER,
                safety_stock INTEGER,
                reorder_point INTEGER,
                stockout_risk_pct REAL,
                recommended_order INTEGER,
                status_label TEXT,
                unit_price REAL,
                expected_revenue REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fh_store ON forecast_history (store_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fh_product ON forecast_history (product_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fh_created ON forecast_history (created_at DESC)")
    conn.close()


def save_forecast_run(data: Dict[str, Any]) -> str:
    """Saves a real forecast execution record to persistent SQLite storage."""
    init_db()
    conn = get_connection()
    
    run_id = f"RUN-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
    inv = data.get("inventory", {})
    fc = data.get("forecast", {})
    features = data.get("features_applied", {})

    with conn:
        conn.execute("""
            INSERT INTO forecast_history (
                run_id, created_at, store_id, product_id, category, target_date,
                forecast_horizon_days, model_used, predicted_units, current_inventory,
                incoming_orders, inventory_position, safety_stock, reorder_point,
                stockout_risk_pct, recommended_order, status_label, unit_price, expected_revenue
            ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            data.get("store_id", "S001"),
            data.get("product_id", "P0001"),
            data.get("category", "Groceries"),
            data.get("target_date", datetime.date.today().strftime("%Y-%m-%d")),
            features.get("forecast_horizon_days", 7),
            data.get("model_used", "LightGBM Regressor (Champion)"),
            fc.get("predicted_units_sold", 0),
            inv.get("current_inventory", 0),
            inv.get("incoming_orders", 0),
            inv.get("inventory_position", 0),
            inv.get("safety_stock", 0),
            inv.get("reorder_point", 0),
            inv.get("stockout_risk_pct", 0.0),
            inv.get("recommended_order_qty", 0),
            inv.get("status_label", "Healthy"),
            features.get("price", 50.0),
            fc.get("expected_revenue", 0.0)
        ))
    conn.close()
    return run_id


def get_history_runs(
    store_id: Optional[str] = None,
    product_id: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """Retrieves paginated and filtered historical forecast records."""
    init_db()
    conn = get_connection()
    
    query = "SELECT * FROM forecast_history WHERE 1=1"
    params = []
    
    if store_id and store_id != "all":
        query += " AND store_id = ?"
        params.append(store_id)
    if product_id and product_id != "all":
        query += " AND product_id = ?"
        params.append(product_id)
    if search:
        query += " AND (run_id LIKE ? OR category LIKE ? OR model_used LIKE ?)"
        s_term = f"%{search}%"
        params.extend([s_term, s_term, s_term])
        
    count_query = f"SELECT COUNT(*) FROM ({query})"
    cursor = conn.execute(count_query, params)
    total_records = cursor.fetchone()[0]
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor = conn.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    
    return {
        "total": total_records,
        "limit": limit,
        "offset": offset,
        "runs": rows
    }


def delete_run(run_id: str) -> bool:
    """Deletes a specific forecast history record."""
    init_db()
    conn = get_connection()
    with conn:
        cursor = conn.execute("DELETE FROM forecast_history WHERE run_id = ?", (run_id,))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def clear_history() -> int:
    """Clears all historical forecast records."""
    init_db()
    conn = get_connection()
    with conn:
        cursor = conn.execute("DELETE FROM forecast_history")
        count = cursor.rowcount
    conn.close()
    return count


# Auto-initialize database schema on load
init_db()
