"""
StockSense AI - Advanced Analytics & Insights Engine
Provides authentic, pre-calculated aggregations directly from the retail dataset:
- Sales Trend & Seasonality Analysis
- Price & Promotion Impact
- Category Demand Breakdown & Market Share
- Multi-SKU Stockout Risk Monitoring & Ranking
- Explainable AI (XAI): Global Feature Importance & Standardized Coefficients
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional

from inventory_engine import InventoryOptimizer


class AnalyticsEngine:
    def __init__(self, data_path: str = "archive/retail_store_inventory.csv", models_dir: str = "models"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.df = None
        self.optimizer = InventoryOptimizer(default_lead_time_days=3, service_level=0.95)
        
        # Load dataset
        if os.path.exists(data_path):
            try:
                self.df = pd.read_csv(data_path)
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                self.df['Month'] = self.df['Date'].dt.month
                self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
                self.df['YearMonth'] = self.df['Date'].dt.to_period('M').astype(str)
            except Exception as e:
                print("AnalyticsEngine: Could not load CSV:", e)
                
        # Load snapshot for inventory alerts
        snapshot_path = os.path.join("data", "sample_inventory_snapshot.csv")
        self.snapshot_df = pd.read_csv(snapshot_path) if os.path.exists(snapshot_path) else pd.DataFrame()
        
        # Load SKU catalogue metadata
        meta_path = os.path.join("data", "store_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def get_sales_trends(self, store_id: Optional[str] = None, category: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes historical sales trends, day-of-week, seasonality, and promotions."""
        if self.df is None or self.df.empty:
            return {"error": "Dataset not available"}

        sub_df = self.df
        if store_id and store_id != "all":
            sub_df = sub_df[sub_df['Store ID'] == store_id]
        if category and category != "all":
            sub_df = sub_df[sub_df['Category'] == category]

        # 1. Monthly Trend
        monthly = sub_df.groupby('YearMonth')['Units Sold'].agg(['sum', 'mean']).reset_index()
        monthly_labels = monthly['YearMonth'].tolist()
        monthly_sales = [int(v) for v in monthly['sum'].tolist()]
        monthly_mean = [round(float(v), 1) for v in monthly['mean'].tolist()]

        # 2. Day of Week Distribution (0=Mon, 6=Sun)
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_agg = sub_df.groupby('DayOfWeek')['Units Sold'].agg(['mean', 'sum']).reindex(range(7)).reset_index()
        dow_labels = [dow_names[i] for i in dow_agg['DayOfWeek']]
        dow_means = [round(float(v), 1) for v in dow_agg['mean'].tolist()]
        dow_totals = [int(v) for v in dow_agg['sum'].tolist()]

        # 3. Seasonality Breakdown
        season_order = ["Spring", "Summer", "Autumn", "Winter"]
        season_agg = sub_df.groupby('Seasonality')['Units Sold'].agg(['mean', 'sum']).reindex(season_order).reset_index()
        season_labels = season_agg['Seasonality'].tolist()
        season_means = [round(float(v), 1) if pd.notnull(v) else 0.0 for v in season_agg['mean'].tolist()]
        season_totals = [int(v) if pd.notnull(v) else 0 for v in season_agg['sum'].tolist()]

        # 4. Holiday / Promotion Comparison
        promo_agg = sub_df.groupby('Holiday/Promotion')['Units Sold'].agg(['count', 'mean', 'sum']).reset_index()
        non_promo_mean = round(float(promo_agg[promo_agg['Holiday/Promotion'] == 0]['mean'].values[0]), 1) if not promo_agg[promo_agg['Holiday/Promotion'] == 0].empty else 0.0
        promo_mean = round(float(promo_agg[promo_agg['Holiday/Promotion'] == 1]['mean'].values[0]), 1) if not promo_agg[promo_agg['Holiday/Promotion'] == 1].empty else 0.0
        promo_uplift_pct = round(((promo_mean - non_promo_mean) / max(1.0, non_promo_mean)) * 100.0, 2)

        # 5. Global Summary Statistics
        total_units = int(sub_df['Units Sold'].sum())
        avg_daily = round(float(sub_df['Units Sold'].mean()), 1)
        daily_sum = sub_df.groupby('Date')['Units Sold'].sum()
        peak_date = daily_sum.idxmax().strftime('%Y-%m-%d')
        peak_volume = int(daily_sum.max())

        return {
            "filters": {"store_id": store_id or "all", "category": category or "all"},
            "summary": {
                "total_units_sold": total_units,
                "average_daily_demand": avg_daily,
                "peak_demand_date": peak_date,
                "peak_daily_units": peak_volume,
                "promo_uplift_pct": promo_uplift_pct,
                "sample_size_rows": len(sub_df)
            },
            "monthly_trend": {
                "labels": monthly_labels,
                "total_sales": monthly_sales,
                "average_sales": monthly_mean
            },
            "day_of_week": {
                "labels": dow_labels,
                "average_sales": dow_means,
                "total_sales": dow_totals
            },
            "seasonality": {
                "labels": season_labels,
                "average_sales": season_means,
                "total_sales": season_totals
            },
            "promotion_comparison": {
                "regular_mean": non_promo_mean,
                "promotion_mean": promo_mean,
                "uplift_pct": promo_uplift_pct
            },
            "academic_notes": "Note: Historical variations by season and promotion reflect empirical sample averages; correlation does not imply causality without controlled experimental intervention."
        }

    def get_price_promotion_impact(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes price elasticity, discount levels, and competitor price differences."""
        if self.df is None or self.df.empty:
            return {"error": "Dataset not available"}

        sub_df = self.df
        if category and category != "all":
            sub_df = sub_df[sub_df['Category'] == category]

        # 1. Sampled Scatter Points (Price vs. Units Sold, 200 stratified rows for fast frontend plotting)
        scatter_sample = sub_df.sample(n=min(250, len(sub_df)), random_state=42)[['Price', 'Units Sold', 'Category', 'Discount']]
        scatter_points = [
            {"x": round(float(r['Price']), 2), "y": int(r['Units Sold']), "cat": r['Category'], "discount": int(r['Discount'])}
            for _, r in scatter_sample.iterrows()
        ]

        # 2. Discount Tier Analysis (0%, 5%, 10%, 15%, 20%)
        discount_agg = sub_df.groupby('Discount')['Units Sold'].agg(['count', 'mean']).reset_index()
        discount_labels = [f"{int(d)}%" for d in discount_agg['Discount']]
        discount_means = [round(float(m), 1) for m in discount_agg['mean']]

        # 3. Competitor Price Gap Impact (Retail Price - Competitor Price)
        sub_df = sub_df.copy()
        sub_df['PriceDiff'] = sub_df['Price'] - sub_df['Competitor Pricing']
        sub_df['CompPosition'] = np.where(sub_df['PriceDiff'] < -1.0, "Cheaper than Competitor",
                                 np.where(sub_df['PriceDiff'] > 1.0, "More Expensive than Competitor", "Similar Price"))
        comp_agg = sub_df.groupby('CompPosition')['Units Sold'].agg(['count', 'mean']).reset_index()
        comp_labels = comp_agg['CompPosition'].tolist()
        comp_means = [round(float(m), 1) for m in comp_agg['mean']]

        # 4. Correlation coefficients
        corr_price = round(float(sub_df['Price'].corr(sub_df['Units Sold'])), 4)
        corr_discount = round(float(sub_df['Discount'].corr(sub_df['Units Sold'])), 4)
        corr_comp = round(float(sub_df['Competitor Pricing'].corr(sub_df['Units Sold'])), 4)

        return {
            "filters": {"category": category or "all"},
            "correlations": {
                "price_to_sales": corr_price,
                "discount_to_sales": corr_discount,
                "competitor_price_to_sales": corr_comp
            },
            "scatter_points": scatter_points,
            "discount_impact": {
                "labels": discount_labels,
                "average_sales": discount_means
            },
            "competitor_pricing_impact": {
                "labels": comp_labels,
                "average_sales": comp_means
            },
            "academic_notes": "Observed price and discount relationships represent statistical correlations within the observed observational retail dataset, not guaranteed causal elasticity."
        }

    def get_category_demand_breakdown(self) -> Dict[str, Any]:
        """Provides high-level category demand totals, revenue, and store distribution."""
        if self.df is None or self.df.empty:
            return {"error": "Dataset not available"}

        cat_agg = self.df.groupby('Category').agg(
            total_units=('Units Sold', 'sum'),
            mean_units=('Units Sold', 'mean'),
            avg_price=('Price', 'mean'),
            record_count=('Units Sold', 'count')
        ).reset_index()

        total_market_units = cat_agg['total_units'].sum()
        cat_agg['market_share_pct'] = round((cat_agg['total_units'] / total_market_units) * 100.0, 1)
        cat_agg['estimated_revenue'] = round(cat_agg['total_units'] * cat_agg['avg_price'], 2)

        cat_names = cat_agg['Category'].tolist()
        units_list = [int(u) for u in cat_agg['total_units']]
        share_list = [float(s) for s in cat_agg['market_share_pct']]
        rev_list = [float(r) for r in cat_agg['estimated_revenue']]

        # Highest and Lowest Demand Categories
        sorted_cats = cat_agg.sort_values(by='total_units', ascending=False)
        highest_cat = {
            "category": sorted_cats.iloc[0]['Category'],
            "units": int(sorted_cats.iloc[0]['total_units']),
            "revenue": float(sorted_cats.iloc[0]['estimated_revenue'])
        }
        lowest_cat = {
            "category": sorted_cats.iloc[-1]['Category'],
            "units": int(sorted_cats.iloc[-1]['total_units']),
            "revenue": float(sorted_cats.iloc[-1]['estimated_revenue'])
        }

        # Store vs Category distribution
        store_cat = self.df.groupby(['Store ID', 'Category'])['Units Sold'].sum().unstack().fillna(0).to_dict(orient='index')

        return {
            "categories": cat_names,
            "total_units": units_list,
            "market_share_pct": share_list,
            "estimated_revenue": rev_list,
            "highest_demand_category": highest_cat,
            "lowest_demand_category": lowest_cat,
            "store_category_matrix": store_cat,
            "total_catalog_units": int(total_market_units)
        }

    def get_stockout_risk_alerts(self, store_id: Optional[str] = None, severity_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates inventory health across all 100 store SKUs.
        Ranks products by risk level and urgency.
        """
        if self.snapshot_df.empty:
            return {"alerts": [], "total_skus": 0}

        sub_snap = self.snapshot_df
        if store_id and store_id != "all":
            sub_snap = sub_snap[sub_snap['Store ID'] == store_id]

        sku_catalogue = self.metadata.get("sku_catalogue", {})
        alerts = []

        for _, row in sub_snap.iterrows():
            s_id = row['Store ID']
            p_id = row['Product ID']
            curr_stock = int(row['Inventory Level'])
            price = float(row['Price'])

            sku_info = sku_catalogue.get(p_id, {})
            p_name = sku_info.get("name", p_id)
            cat = sku_info.get("category", "Groceries")
            hist_std = float(sku_info.get("historical_std_sales", 25.0))
            hist_mean = float(sku_info.get("historical_mean_sales", 136.0))

            # Run inventory diagnostic
            inv_diag = self.optimizer.evaluate_inventory_health(
                current_inventory=curr_stock,
                predicted_daily_demand=hist_mean,
                std_daily_demand=hist_std,
                incoming_orders=0,
                lead_time_days=3,
                service_level=0.95
            )

            severity = inv_diag["severity"]
            if severity_filter and severity_filter != "all" and severity != severity_filter:
                continue

            # Urgency score: higher risk + smaller current stock relative to ROP
            urgency_score = round(inv_diag["stockout_risk_pct"] * 1.5 + max(0, inv_diag["reorder_point"] - curr_stock), 1)

            alerts.append({
                "store_id": s_id,
                "product_id": p_id,
                "product_name": p_name,
                "category": cat,
                "price": price,
                "current_inventory": curr_stock,
                "incoming_orders": 0,
                "inventory_position": curr_stock,
                "reorder_point": inv_diag["reorder_point"],
                "safety_stock": inv_diag["safety_stock"],
                "stockout_risk_pct": inv_diag["stockout_risk_pct"],
                "recommended_order_qty": inv_diag["recommended_order_qty"],
                "status": inv_diag["status"],
                "status_label": inv_diag["status_label"],
                "severity": severity,
                "urgency_score": urgency_score,
                "reason": f"Inventory position ({curr_stock} units) is { 'below' if curr_stock <= inv_diag['reorder_point'] else 'above'} ROP ({inv_diag['reorder_point']} units) with {inv_diag['stockout_risk_pct']}% stockout risk during 3-day lead time.",
                "action": inv_diag["action_recommendation"]
            })

        # Sort by urgency descending
        alerts.sort(key=lambda x: (0 if x["severity"] == "danger" else 1 if x["severity"] == "warning" else 2, -x["urgency_score"]))

        # Count severities
        danger_count = sum(1 for a in alerts if a["severity"] == "danger")
        warning_count = sum(1 for a in alerts if a["severity"] == "warning")
        success_count = sum(1 for a in alerts if a["severity"] == "success")
        info_count = sum(1 for a in alerts if a["severity"] == "info")

        return {
            "total_items": len(alerts),
            "summary_counts": {
                "critical": danger_count,
                "reorder_needed": warning_count,
                "optimal": success_count,
                "overstocked": info_count
            },
            "alerts": alerts
        }

    def get_model_explainability(self) -> Dict[str, Any]:
        """
        Extracts global feature importances from LightGBM and standardized coefficients from Linear Regression.
        """
        champion_path = os.path.join(self.models_dir, "stocksense_pipeline.joblib")
        lr_path = os.path.join(self.models_dir, "stocksense_linear_regression_pipeline.joblib")

        lgb_importances = []
        lr_coefficients = []

        # 1. LightGBM Tree Feature Importance
        if os.path.exists(champion_path):
            try:
                pipe = joblib.load(champion_path)
                preprocessor = pipe.named_steps.get('preprocessor')
                regressor = pipe.named_steps.get('regressor')
                
                if preprocessor and regressor and hasattr(regressor, 'feature_importances_'):
                    feature_names = preprocessor.get_feature_names_out()
                    raw_imp = regressor.feature_importances_
                    # Group one-hot features back to core domain features
                    feature_dict = {}
                    for name, imp in zip(feature_names, raw_imp):
                        clean_name = name.split('__')[-1].split('_')[0]
                        feature_dict[clean_name] = feature_dict.get(clean_name, 0.0) + float(imp)
                    
                    total_imp = sum(feature_dict.values())
                    lgb_importances = [
                        {"feature": k, "importance_score": round(v, 2), "share_pct": round((v / max(1.0, total_imp)) * 100.0, 1)}
                        for k, v in sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
                    ]
            except Exception as e:
                print("LGB feature importance extraction error:", e)

        # 2. Linear Regression Coefficients
        if os.path.exists(lr_path):
            try:
                pipe = joblib.load(lr_path)
                preprocessor = pipe.named_steps.get('preprocessor')
                regressor = pipe.named_steps.get('regressor')
                
                if preprocessor and regressor and hasattr(regressor, 'coef_'):
                    feature_names = preprocessor.get_feature_names_out()
                    coefs = regressor.coef_
                    coef_dict = {}
                    for name, c in zip(feature_names, coefs):
                        clean_name = name.split('__')[-1]
                        coef_dict[clean_name] = round(float(c), 3)
                    
                    # Top positive and negative coefficients
                    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                    lr_coefficients = [{"feature": k, "coefficient": v} for k, v in sorted_coefs]
            except Exception as e:
                print("LR coefficients extraction error:", e)

        return {
            "lightgbm_feature_importance": lgb_importances,
            "linear_regression_top_coefficients": lr_coefficients,
            "interpretation_guide": {
                "tree_importance": "Gini/split feature importance identifies which variables the decision trees split on most frequently to minimize mean squared error.",
                "linear_coefficients": "Standardized coefficients show the expected change in predicted Units Sold per standard deviation change in the input variable, holding all other features fixed.",
                "academic_warning": "Neither tree feature importances nor linear regression weights establish causal direction. They represent statistical associations observed within the training dataset."
            }
        }


# Singleton analytics engine
_analytics_instance = None

def get_analytics():
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = AnalyticsEngine()
    return _analytics_instance
