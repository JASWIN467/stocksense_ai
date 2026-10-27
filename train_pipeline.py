"""
StockSense AI - Complete Training & Model Persistence Pipeline
Executes chronological time-aware train-test splits, computes college benchmark
metrics (MAE, RMSE, R2, WAPE) across naive baselines and ML regressors, and exports
a production-ready Scikit-Learn pipeline for instant inference.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb


def calculate_wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error (WAPE)."""
    return float((np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100.0)


def evaluate_predictions(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    w = float(calculate_wape(y_true, y_pred))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "WAPE_pct": round(w, 2)
    }


def main():
    print("==================================================================")
    print("  StockSense AI - End-to-End Demand Forecasting Pipeline Trainer  ")
    print("==================================================================")

    data_path = os.path.join("archive", "retail_store_inventory.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"\n[1/6] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"      Total records loaded: {len(df):,} rows x {df.shape[1]} columns")

    # Parse and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Store ID', 'Product ID']).reset_index(drop=True)
    
    date_min = df['Date'].min().strftime('%Y-%m-%d')
    date_max = df['Date'].max().strftime('%Y-%m-%d')
    print(f"      Date Range: {date_min} to {date_max} (731 days)")

    # Temporal feature engineering
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Day'] = df['Date'].dt.day

    # Feature definitions
    num_cols = ['Inventory Level', 'Price', 'Discount', 'Competitor Pricing', 'Month', 'DayOfWeek', 'Day']
    cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Holiday/Promotion', 'Seasonality']
    all_features = num_cols + cat_cols
    target = 'Units Sold'

    # Chronological 80/20 Time-Aware Split
    unique_dates = df['Date'].sort_values().unique()
    split_idx = int(len(unique_dates) * 0.8)
    cutoff_date = unique_dates[split_idx]
    
    train_mask = df['Date'] < cutoff_date
    test_mask = df['Date'] >= cutoff_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"\n[2/6] Chronological Time-Aware Split established:")
    print(f"      Cutoff Date: {pd.to_datetime(cutoff_date).strftime('%Y-%m-%d')}")
    print(f"      Train Set: {len(train_df):,} rows ({train_mask.mean()*100:.1f}%)")
    print(f"      Test Set:  {len(test_df):,} rows ({test_mask.mean()*100:.1f}%)")

    X_train = train_df[all_features]
    y_train = train_df[target]
    X_test = test_df[all_features]
    y_test = test_df[target]

    # Preprocessor definition
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    metrics_summary = {}

    # [3/6] Naive Baselines
    print("\n[3/6] Computing Naive Baselines for rigorous academic benchmarking...")
    
    # Baseline A: Target Mean
    y_mean_pred = np.full(len(y_test), y_train.mean())
    metrics_summary["Naive: Target Mean"] = evaluate_predictions(y_test, y_mean_pred)
    
    # Baseline B: Lag-1 (Yesterday's sales per SKU)
    df_sorted_sku = df.sort_values(by=['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    df_sorted_sku['Lag1'] = df_sorted_sku.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(1)
    df_sorted_sku['Rolling7'] = df_sorted_sku.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df_sorted_sku['ExpSmooth'] = df_sorted_sku.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
        lambda s: s.shift(1).ewm(alpha=0.2, adjust=False).mean()
    )
    
    test_sku_df = df_sorted_sku[df_sorted_sku['Date'] >= cutoff_date].dropna(subset=['Lag1', 'Rolling7', 'ExpSmooth'])
    metrics_summary["Naive: Yesterday Sales (Lag-1)"] = evaluate_predictions(test_sku_df['Units Sold'], test_sku_df['Lag1'])
    metrics_summary["Naive: 7-Day Rolling Average"] = evaluate_predictions(test_sku_df['Units Sold'], test_sku_df['Rolling7'])
    metrics_summary["Time-Series: Exponential Smoothing (Alpha=0.2)"] = evaluate_predictions(test_sku_df['Units Sold'], test_sku_df['ExpSmooth'])

    # [4/6] Machine Learning Regressors
    print("\n[4/6] Training Machine Learning Regressors...")

    # Model 1: Linear Regression (The User's Core Baseline Model)
    print("      -> Training Linear Regression (Honest Baseline)...")
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    p_lr = lr_pipe.predict(X_test)
    metrics_summary["Linear Regression (Baseline)"] = evaluate_predictions(y_test, p_lr)

    # Model 2: Ridge Regression (L2 Regularized)
    print("      -> Training Ridge Regression (Alpha=10.0)...")
    ridge_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=10.0))
    ])
    ridge_pipe.fit(X_train, y_train)
    p_ridge = ridge_pipe.predict(X_test)
    metrics_summary["Ridge Regression"] = evaluate_predictions(y_test, p_ridge)

    # Model 3: LightGBM Regressor (Non-linear Gradient Boosted Champion)
    print("      -> Training LightGBM Regressor (Champion)...")
    lgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        ))
    ])
    lgb_pipe.fit(X_train, y_train)
    p_lgb = lgb_pipe.predict(X_test)
    metrics_summary["LightGBM Regressor (Champion)"] = evaluate_predictions(y_test, p_lgb)

    # Model 4: Random Forest Regressor
    rf_path = os.path.join("models", "stocksense_rf_pipeline.joblib")
    if os.path.exists(rf_path):
        print("      -> Loading pre-trained Random Forest Regressor...")
        rf_pipe = joblib.load(rf_path)
    else:
        print("      -> Training Random Forest Regressor (50 Trees)...")
        from sklearn.ensemble import RandomForestRegressor
        rf_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1))
        ])
        rf_pipe.fit(X_train, y_train)
        joblib.dump(rf_pipe, rf_path)
    p_rf = rf_pipe.predict(X_test)
    metrics_summary["Random Forest Regressor"] = evaluate_predictions(y_test, p_rf)

    # Documenting the Legacy Model (Target = Demand Forecast with Units Sold Leakage)
    X_legacy = df[['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Holiday/Promotion', 'Competitor Pricing']]
    y_legacy = df['Demand Forecast']
    X_leg_tr, X_leg_te, y_leg_tr, y_leg_te = train_test_split(X_legacy, y_legacy, test_size=0.2, random_state=42)
    legacy_lr = LinearRegression().fit(X_leg_tr, y_leg_tr)
    p_leg = legacy_lr.predict(X_leg_te)
    metrics_summary["Legacy Model (Demand Forecast Target, Leaked Units Sold)"] = evaluate_predictions(y_leg_te, p_leg)

    print("\n" + "="*70)
    print("  COLLEGE BENCHMARK EVALUATION MATRIX (Honest Chronological Split)")
    print("="*70)
    print(f"{'Model / Architecture':<40} | {'MAE':<8} | {'RMSE':<8} | {'R2':<8} | {'WAPE':<8}")
    print("-"*70)
    for model_name, m in metrics_summary.items():
        print(f"{model_name:<40} | {m['MAE']:<8.2f} | {m['RMSE']:<8.2f} | {m['R2']:<8.4f} | {m['WAPE_pct']:<7.2f}%")
    print("="*70)

    # [5/6] Exporting Artifacts
    print("\n[5/6] Exporting production-ready ML artifacts...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save the LightGBM Champion Pipeline
    pipeline_path = os.path.join("models", "stocksense_pipeline.joblib")
    joblib.dump(lgb_pipe, pipeline_path)
    print(f"      -> Champion pipeline saved to: {pipeline_path}")

    # Also save the Honest Linear Regression Pipeline
    lr_pipeline_path = os.path.join("models", "stocksense_linear_regression_pipeline.joblib")
    joblib.dump(lr_pipe, lr_pipeline_path)
    print(f"      -> Baseline LR pipeline saved to: {lr_pipeline_path}")

    # Save benchmark metrics JSON
    metrics_path = os.path.join("models", "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"      -> Benchmark metrics saved to: {metrics_path}")

    # [6/6] Generating metadata & latest inventory snapshot for the UI
    print("\n[6/6] Generating store metadata and latest inventory snapshot...")
    
    stores = sorted(df['Store ID'].unique().tolist())
    products = sorted(df['Product ID'].unique().tolist())
    categories = sorted(df['Category'].unique().tolist())
    regions = sorted(df['Region'].unique().tolist())
    weathers = sorted(df['Weather Condition'].unique().tolist())
    seasons = sorted(df['Seasonality'].unique().tolist())
    
    # Build SKU catalogue mapping
    sku_catalogue = {}
    for p in products:
        p_row = df[df['Product ID'] == p].iloc[0]
        sku_catalogue[p] = {
            "name": f"Product {p[-4:]}",
            "category": p_row['Category'],
            "base_price": round(float(p_row['Price']), 2),
            "historical_mean_sales": round(float(df[df['Product ID'] == p]['Units Sold'].mean()), 1),
            "historical_std_sales": round(float(df[df['Product ID'] == p]['Units Sold'].std()), 1)
        }

    metadata = {
        "stores": stores,
        "products": products,
        "categories": categories,
        "regions": regions,
        "weather_conditions": weathers,
        "seasons": seasons,
        "sku_catalogue": sku_catalogue,
        "dataset_rows": len(df),
        "date_range": [date_min, date_max],
        "metrics": metrics_summary
    }

    metadata_path = os.path.join("data", "store_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"      -> Store metadata saved to: {metadata_path}")

    # Latest inventory snapshot (most recent date in dataset: 2024-01-01)
    latest_date = df['Date'].max()
    snapshot = df[df['Date'] == latest_date].copy()
    snapshot_path = os.path.join("data", "sample_inventory_snapshot.csv")
    snapshot.to_csv(snapshot_path, index=False)
    print(f"      -> Inventory snapshot (Date: {latest_date.strftime('%Y-%m-%d')}) saved to: {snapshot_path}")

    print("\n>>> Pipeline Training & Export Completed Successfully! <<<")


if __name__ == "__main__":
    main()
