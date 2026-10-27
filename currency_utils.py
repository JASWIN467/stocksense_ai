"""
StockSense AI - Indian Rupee (INR ₹) Currency Formatting Module
Implements the official Indian numbering system digit grouping:
- 1,000      -> ₹1,000
- 1,00,000   -> ₹1,00,000 (1 Lakh)
- 10,00,000  -> ₹10,00,000 (10 Lakhs)
- 1,00,00,000-> ₹1,00,00,000 (1 Crore)

Currency Assumption:
The raw retail inventory dataset contains unit prices and competitor pricing as numeric values.
In alignment with enterprise Indian market standards, all monetary amounts (prices, holding costs,
ordering costs, and revenues) are treated as Indian Rupees (INR ₹) under the Indian numbering system.
"""

from typing import Union


def format_inr(val: Union[int, float, str], decimals: int = 2) -> str:
    """
    Formats a numeric value into Indian Rupees (INR ₹) using the Indian numbering system.
    Examples:
        format_inr(1000, 0)     -> '₹1,000'
        format_inr(100000, 0)   -> '₹1,00,000'
        format_inr(1000000, 0)  -> '₹10,00,000'
        format_inr(10000000, 0) -> '₹1,00,00,000'
        format_inr(6502.5, 2)   -> '₹6,502.50'
    """
    if val is None or val == "":
        return "₹0.00" if decimals > 0 else "₹0"

    try:
        num = float(val)
    except (ValueError, TypeError):
        return f"₹{val}"

    is_negative = num < 0
    num = abs(num)

    if decimals > 0:
        base_str = f"{num:.{decimals}f}"
        parts = base_str.split(".")
        int_part, dec_part = parts[0], "." + parts[1]
    else:
        int_part = str(int(round(num)))
        dec_part = ""

    # Indian numbering grouping: rightmost 3 digits, then pairs of 2 digits
    if len(int_part) <= 3:
        formatted_int = int_part
    else:
        last3 = int_part[-3:]
        remaining = int_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        formatted_int = ",".join(reversed(groups)) + "," + last3

    prefix = "-₹" if is_negative else "₹"
    return f"{prefix}{formatted_int}{dec_part}"
