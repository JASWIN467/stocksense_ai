"""
StockSense AI - Operations Research & Inventory Optimization Engine
Authoritative backend for continuous-review (s, Q) and periodic-review (R, S) policies:
- Safety Stock (SS) with normal distribution Z-factors
- Reorder Point (ROP) calculation
- Stockout Risk Probability via Normal Cumulative Distribution Function (CDF)
- Economic Order Quantity (EOQ)
- Inventory Position accounting for Current Inventory + Incoming Orders
"""

import math
from typing import Dict, Any, Optional


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution N(0, 1)."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def z_score_from_service_level(service_level: float) -> float:
    """
    Returns standard normal quantile Z for a given Cycle Service Level (CSL).
    Uses high-precision rational approximation (Abramowitz and Stegun).
    """
    sl = max(0.50, min(0.9999, float(service_level)))
    # Well-known exact lookups for standard retail service levels
    discrete_map = {
        0.90: 1.2816,
        0.95: 1.6449,
        0.98: 2.0537,
        0.99: 2.3263
    }
    for key, val in discrete_map.items():
        if abs(sl - key) < 0.005:
            return val
    
    # Rational approximation for inverse normal CDF
    # For p >= 0.5:
    p = sl
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    z = t - ((c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
    return round(z, 4)


class InventoryOptimizer:
    def __init__(self, default_lead_time_days: int = 3, service_level: float = 0.95):
        """
        :param default_lead_time_days: Supplier lead time (L) in days.
        :param service_level: Target Cycle Service Level (CSL), e.g. 0.95 for 95%.
        """
        self.lead_time = max(1, int(default_lead_time_days))
        self.service_level = max(0.50, min(0.999, float(service_level)))
        self.z_score = z_score_from_service_level(self.service_level)

    def calculate_safety_stock(
        self,
        std_daily_demand: float,
        lead_time_days: Optional[int] = None,
        service_level: Optional[float] = None,
        lead_time_std_days: float = 0.0,
        mean_daily_demand: Optional[float] = None
    ) -> int:
        """
        Safety Stock (SS):
        - Deterministic Lead Time (sigma_L = 0):
            SS = Z * sigma_d * sqrt(L)
        - Stochastic Lead Time (sigma_L > 0):
            SS = Z * sqrt(L * sigma_d^2 + (d^2 * sigma_L^2))
        Protects against both demand variability and supplier replenishment delays.
        """
        L = max(1, int(lead_time_days if lead_time_days is not None else self.lead_time))
        z = z_score_from_service_level(service_level) if service_level is not None else self.z_score
        sigma_d = max(0.5, float(std_daily_demand))
        sigma_L = max(0.0, float(lead_time_std_days))

        if sigma_L > 0.0 and mean_daily_demand is not None and mean_daily_demand > 0:
            d = float(mean_daily_demand)
            total_lead_time_var = (L * (sigma_d ** 2)) + ((d ** 2) * (sigma_L ** 2))
            ss = z * math.sqrt(max(0.1, total_lead_time_var))
        else:
            ss = z * sigma_d * math.sqrt(L)
        return max(1, int(math.ceil(ss)))

    def calculate_reorder_point(
        self,
        predicted_daily_demand: float,
        safety_stock: int,
        lead_time_days: Optional[int] = None
    ) -> int:
        """
        Reorder Point (ROP) = (Expected Daily Demand * L) + Safety Stock
        The continuous inventory threshold that triggers a replenishment order.
        """
        L = max(1, int(lead_time_days if lead_time_days is not None else self.lead_time))
        d = max(0.0, float(predicted_daily_demand))
        expected_lead_time_demand = d * L
        rop = expected_lead_time_demand + safety_stock
        return max(1, int(math.ceil(rop)))

    def calculate_stockout_risk(
        self,
        inventory_position: int,
        predicted_daily_demand: float,
        std_daily_demand: float,
        lead_time_days: Optional[int] = None,
        lead_time_std_days: float = 0.0
    ) -> float:
        """
        Computes the calibrated probability of stocking out during lead time:
        P(Demand_L > Inventory_Position) = 1 - Phi((Inventory_Position - mu_L) / sigma_LT)
        Where:
            mu_L = predicted_daily_demand * L
            sigma_LT = sqrt(L * sigma_d^2 + (d^2 * sigma_L^2))
            Phi = Standard Normal Cumulative Distribution Function
        """
        L = max(1, int(lead_time_days if lead_time_days is not None else self.lead_time))
        d = max(0.0, float(predicted_daily_demand))
        mu_l = d * L
        sigma_d = max(0.5, float(std_daily_demand))
        sigma_L = max(0.0, float(lead_time_std_days))

        if sigma_L > 0.0 and d > 0:
            total_var = (L * (sigma_d ** 2)) + ((d ** 2) * (sigma_L ** 2))
            sigma_lt = max(1.0, math.sqrt(total_var))
        else:
            sigma_lt = max(1.0, sigma_d * math.sqrt(L))

        if inventory_position <= 0:
            return 100.0

        z = (float(inventory_position) - mu_l) / sigma_lt
        prob_no_stockout = norm_cdf(z)
        risk_pct = max(0.0, min(100.0, (1.0 - prob_no_stockout) * 100.0))
        return round(risk_pct, 1)

    def evaluate_inventory_health(
        self,
        current_inventory: int,
        predicted_daily_demand: float,
        std_daily_demand: float = 25.0,
        incoming_orders: int = 0,
        lead_time_days: Optional[int] = None,
        service_level: Optional[float] = None,
        forecast_horizon_days: int = 7,
        holding_cost_unit: float = 2.0,
        order_setup_cost: float = 50.0,
        lead_time_std_days: float = 0.0
    ) -> Dict[str, Any]:
        """
        Authoritative single-source inventory policy evaluation for a specific SKU.
        Accurately factors in:
        - Current On-Hand Inventory
        - Incoming Open Purchase Orders
        - True Inventory Position = Current Stock + Incoming Orders
        - Lead Time (L) and Review Horizon (R)
        - Service Level (CSL) and Safety Stock (SS)
        - Reorder Point (ROP) & Order-Up-To Level (S)
        """
        L = max(1, int(lead_time_days if lead_time_days is not None else self.lead_time))
        csl = max(0.50, min(0.999, float(service_level if service_level is not None else self.service_level)))
        z = z_score_from_service_level(csl)
        horizon = max(1, int(forecast_horizon_days))

        d = max(0.1, float(predicted_daily_demand))
        curr_stock = max(0, int(current_inventory))
        incoming = max(0, int(incoming_orders))

        # Effective Inventory Position
        inventory_position = curr_stock + incoming

        # Safety Stock and ROP (incorporating stochastic lead time variance if provided)
        safety_stock = self.calculate_safety_stock(
            std_daily_demand=std_daily_demand,
            lead_time_days=L,
            service_level=csl,
            lead_time_std_days=lead_time_std_days,
            mean_daily_demand=d
        )
        rop = self.calculate_reorder_point(d, safety_stock, L)

        # Order-Up-To Target (S): covers lead time plus review horizon plus buffer
        order_up_to = int(math.ceil((d * (L + horizon)) + safety_stock))

        # Economic Order Quantity (EOQ)
        annual_demand = d * 365.0
        h = max(0.5, float(holding_cost_unit))
        s_cost = max(10.0, float(order_setup_cost))
        eoq = int(math.ceil(math.sqrt((2.0 * annual_demand * s_cost) / h)))

        # Stockout Risk during lead time (based on inventory position)
        stockout_risk = self.calculate_stockout_risk(
            inventory_position=inventory_position,
            predicted_daily_demand=d,
            std_daily_demand=std_daily_demand,
            lead_time_days=L,
            lead_time_std_days=lead_time_std_days
        )

        # Replenishment Recommendation Logic
        # Recommended Order Quantity = max(0, Target Level - Inventory Position)
        if inventory_position <= rop:
            recommended_order = max(eoq, order_up_to - inventory_position)
        else:
            recommended_order = 0

        # Health Classification & Action Recommendation
        if curr_stock == 0 and incoming == 0:
            status = "CRITICAL_STOCKOUT"
            status_label = "Critical Stockout"
            severity = "danger"
            action = f"EMERGENCY ORDER: Shelf is empty. Immediate replenishment of {recommended_order} units required."
        elif inventory_position <= int(rop * 0.5):
            status = "HIGH_STOCKOUT_RISK"
            status_label = "Severe Understock"
            severity = "danger"
            action = f"URGENT REORDER: Inventory position ({inventory_position} units) is critically below ROP ({rop} units). Recommend ordering {recommended_order} units."
        elif inventory_position <= rop:
            status = "REORDER_TRIGGERED"
            status_label = "Reorder Point Reached"
            severity = "warning"
            action = f"TRIGGER ORDER: Inventory position ({inventory_position} units) reached ROP ({rop} units). Place order for {recommended_order} units."
        elif inventory_position > int(rop * 2.8):
            status = "OVERSTOCKED"
            status_label = "Overstocked"
            severity = "info"
            recommended_order = 0
            excess = inventory_position - int(rop * 1.8)
            action = f"EXCESS INVENTORY: Holding ~{excess} excess units above target. Pause replenishment; consider targeted promotion."
        else:
            status = "HEALTHY"
            status_label = "Optimal Stock"
            severity = "success"
            recommended_order = 0
            action = f"OPTIMAL STOCK: Inventory level is sufficient for the next {round(inventory_position / d, 1)} days of demand."

        days_of_supply = round(inventory_position / d, 1) if d > 0 else 999.0

        return {
            "current_inventory": curr_stock,
            "incoming_orders": incoming,
            "inventory_position": inventory_position,
            "predicted_daily_demand": round(d, 1),
            "safety_stock": safety_stock,
            "reorder_point": rop,
            "order_up_to_level": order_up_to,
            "economic_order_qty": eoq,
            "recommended_order_qty": recommended_order,
            "stockout_risk_pct": stockout_risk,
            "days_of_supply": days_of_supply,
            "lead_time_days": L,
            "lead_time_std_days": round(float(lead_time_std_days), 2),
            "forecast_horizon_days": horizon,
            "service_level_pct": int(round(csl * 100)),
            "z_score": z,
            "status": status,
            "status_label": status_label,
            "severity": severity,
            "action_recommendation": action
        }
