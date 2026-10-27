"""
StockSense AI - Professional PDF Report Generation Module
Generates publication-quality PDF reports using ReportLab for academic and enterprise audits:
1. Single SKU Forecast Report
2. Inventory Recommendation Report
3. Model Benchmark & Evaluation Report
4. Category Demand Analytics Report
5. Combined Academic Project Summary Report
"""

import os
import io
import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from currency_utils import format_inr

# TrueType font registration for Indian Rupee (₹) symbol support
FONT_NORMAL = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"

try:
    if os.path.exists("C:/Windows/Fonts/segoeui.ttf") and os.path.exists("C:/Windows/Fonts/segoeuib.ttf"):
        pdfmetrics.registerFont(TTFont("SegoeUI", "C:/Windows/Fonts/segoeui.ttf"))
        pdfmetrics.registerFont(TTFont("SegoeUI-Bold", "C:/Windows/Fonts/segoeuib.ttf"))
        if os.path.exists("C:/Windows/Fonts/segoeuii.ttf"):
            pdfmetrics.registerFont(TTFont("SegoeUI-Italic", "C:/Windows/Fonts/segoeuii.ttf"))
            FONT_ITALIC = "SegoeUI-Italic"
        else:
            FONT_ITALIC = "SegoeUI"
        FONT_NORMAL = "SegoeUI"
        FONT_BOLD = "SegoeUI-Bold"
    elif os.path.exists("C:/Windows/Fonts/arial.ttf") and os.path.exists("C:/Windows/Fonts/arialbd.ttf"):
        pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
        pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
        if os.path.exists("C:/Windows/Fonts/ariali.ttf"):
            pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
            FONT_ITALIC = "Arial-Italic"
        else:
            FONT_ITALIC = "Arial"
        FONT_NORMAL = "Arial"
        FONT_BOLD = "Arial-Bold"
except Exception as e:
    print("Warning registering TrueType font for PDF:", e)


def create_base_styles():
    styles = getSampleStyleSheet()
    
    # Custom Brand Styles with INR ₹ Font Support
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName=FONT_BOLD,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName=FONT_NORMAL,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName=FONT_BOLD,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName=FONT_NORMAL,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    badge_style = ParagraphStyle(
        'DocBadge',
        parent=styles['Normal'],
        fontName=FONT_BOLD,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2563eb')
    )

    note_style = ParagraphStyle(
        'DocNote',
        parent=styles['Normal'],
        fontName=FONT_ITALIC,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#475569')
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "h2": h2_style,
        "body": body_style,
        "badge": badge_style,
        "note": note_style
    }


def add_header(story, title: str, subtitle: str, styles):
    story.append(Paragraph("StockSense AI — Intelligent Demand & Inventory Analytics", styles["badge"]))
    story.append(Paragraph(title, styles["title"]))
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"{subtitle} | Generated: {timestamp_str} | Dataset: Kaggle Retail (2022-2024)", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=12))


def generate_forecast_pdf(data: Dict[str, Any]) -> io.BytesIO:
    """Generates Single SKU Demand Forecast & Inventory Diagnostic PDF."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = create_base_styles()
    story = []

    p_name = data.get("product_name", data.get("product_id"))
    store_id = data.get("store_id", "S001")
    add_header(story, f"SKU Demand Forecast: {p_name}", f"Store {store_id} ({data.get('region', 'North')} Region)", styles)

    # Key Metrics Table
    fc = data.get("forecast", {})
    inv = data.get("inventory", {})
    ci = fc.get("confidence_interval_95", {})

    metrics_data = [
        [
            Paragraph("<b>Predicted Daily Demand:</b>", styles["body"]),
            Paragraph(f"<b>{fc.get('predicted_units_sold', 0)} Units</b>", styles["body"]),
            Paragraph("<b>Current On-Hand Stock:</b>", styles["body"]),
            Paragraph(f"{inv.get('current_inventory', 0)} Units", styles["body"])
        ],
        [
            Paragraph("<b>95% Confidence Interval:</b>", styles["body"]),
            Paragraph(f"[{ci.get('lower', 0)} – {ci.get('upper', 0)}] Units", styles["body"]),
            Paragraph("<b>Incoming Orders:</b>", styles["body"]),
            Paragraph(f"{inv.get('incoming_orders', 0)} Units", styles["body"])
        ],
        [
            Paragraph("<b>Safety Stock Buffer (SS):</b>", styles["body"]),
            Paragraph(f"{inv.get('safety_stock', 0)} Units (CSL {inv.get('service_level_pct', 95)}%)", styles["body"]),
            Paragraph("<b>Reorder Point (ROP):</b>", styles["body"]),
            Paragraph(f"<b>{inv.get('reorder_point', 0)} Units</b>", styles["body"])
        ],
        [
            Paragraph("<b>Stockout Risk Probability:</b>", styles["body"]),
            Paragraph(f"<b>{inv.get('stockout_risk_pct', 0.0)}%</b>", styles["body"]),
            Paragraph("<b>Recommended Order:</b>", styles["body"]),
            Paragraph(f"<b>{inv.get('recommended_order_qty', 0)} Units</b>", styles["body"])
        ],
        [
            Paragraph("<b>Health Classification:</b>", styles["body"]),
            Paragraph(f"<b>{inv.get('status_label', 'Healthy')}</b>", styles["body"]),
            Paragraph("<b>Model Utilized:</b>", styles["body"]),
            Paragraph(f"{data.get('model_used', 'LightGBM Regressor')}", styles["body"])
        ]
    ]

    t = Table(metrics_data, colWidths=[140, 130, 140, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Action Recommendation Card
    story.append(Paragraph("Operations Action Recommendation", styles["h2"]))
    rec_text = inv.get("action_recommendation", "Maintain scheduled inventory replenishment.")
    rec_table = Table([[Paragraph(f"<b>Recommendation:</b> {rec_text}", styles["body"])]], colWidths=[540])
    rec_bg = colors.HexColor('#fee2e2') if inv.get("severity") == "danger" else (colors.HexColor('#fef3c7') if inv.get("severity") == "warning" else colors.HexColor('#dcfce7'))
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), rec_bg),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 14))

    # Applied Scenario Inputs
    story.append(Paragraph("Input Scenario Features & Assumptions", styles["h2"]))
    feats = data.get("features_applied", {})
    feat_data = [
        [
            Paragraph(f"<b>Unit Price:</b> {format_inr(feats.get('price', 0), 2)}", styles["body"]),
            Paragraph(f"<b>Promotional Discount:</b> {feats.get('discount_pct', 0)}%", styles["body"]),
            Paragraph(f"<b>Competitor Price:</b> {format_inr(feats.get('competitor_price', 0), 2)}", styles["body"])
        ],
        [
            Paragraph(f"<b>Weather:</b> {feats.get('weather', 'Sunny')}", styles["body"]),
            Paragraph(f"<b>Holiday/Promotion:</b> {'Yes' if feats.get('holiday_promotion') else 'No'}", styles["body"]),
            Paragraph(f"<b>Seasonality:</b> {feats.get('seasonality', 'Summer')}", styles["body"])
        ],
        [
            Paragraph(f"<b>Lead Time (L):</b> {feats.get('lead_time_days', 3)} Days", styles["body"]),
            Paragraph(f"<b>Service Level:</b> {int(feats.get('service_level', 0.95)*100)}%", styles["body"]),
            Paragraph(f"<b>Forecast Horizon:</b> {feats.get('forecast_horizon_days', 7)} Days", styles["body"])
        ]
    ]
    ft = Table(feat_data, colWidths=[180, 180, 180])
    ft.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(ft)
    story.append(Spacer(1, 16))

    # Academic & Operational Disclaimers
    story.append(Paragraph("Mathematical Formulations & Operational Assumptions", styles["h2"]))
    story.append(Paragraph("1. <b>Safety Stock:</b> SS = Z * σ_d * √L, where Z is the normal service factor (Z=1.65 for 95% CSL).<br/>"
                           "2. <b>Reorder Point:</b> ROP = (Predicted Daily Demand * L) + SS.<br/>"
                           "3. <b>Order Recommendation:</b> Q = max(0, S - (On-Hand + Incoming Orders)). This is an advisory recommendation and does not submit an automated purchase order to suppliers.<br/>"
                           "4. <b>Confidence Interval:</b> Derived from out-of-sample chronological evaluation RMSE.<br/>"
                           "5. <b>Currency Assumption:</b> Source dataset monetary values are evaluated directly in Indian Rupees (INR ₹) under the Indian numbering system.", styles["note"]))

    doc.build(story)
    buf.seek(0)
    return buf


def generate_inventory_recommendation_pdf(store_id: str, matrix_data: List[Dict[str, Any]]) -> io.BytesIO:
    """Generates Full Store Inventory Recommendations PDF Report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
    styles = create_base_styles()
    story = []

    add_header(story, f"Inventory Replenishment Recommendations: Store {store_id}", "Continuous Review (s, Q) Multi-SKU Replenishment Matrix", styles)

    # Matrix Table Header
    table_rows = [
        [
            Paragraph("<b>SKU</b>", styles["body"]),
            Paragraph("<b>Category</b>", styles["body"]),
            Paragraph("<b>Stock</b>", styles["body"]),
            Paragraph("<b>Pred.</b>", styles["body"]),
            Paragraph("<b>SS</b>", styles["body"]),
            Paragraph("<b>ROP</b>", styles["body"]),
            Paragraph("<b>Risk %</b>", styles["body"]),
            Paragraph("<b>Order Qty</b>", styles["body"]),
            Paragraph("<b>Status</b>", styles["body"])
        ]
    ]

    for item in matrix_data[:20]: # 20 products
        status = item.get("status_label", "Healthy")
        risk = item.get("stockout_risk_pct", 0.0)
        table_rows.append([
            Paragraph(f"{item.get('product_id')}", styles["body"]),
            Paragraph(f"{item.get('category')[:8]}", styles["body"]),
            Paragraph(f"{item.get('current_inventory')}", styles["body"]),
            Paragraph(f"{item.get('predicted_demand')}", styles["body"]),
            Paragraph(f"{item.get('safety_stock')}", styles["body"]),
            Paragraph(f"{item.get('reorder_point')}", styles["body"]),
            Paragraph(f"{risk}%", styles["body"]),
            Paragraph(f"<b>{item.get('recommended_order')}</b>", styles["body"]),
            Paragraph(f"{status[:12]}", styles["body"])
        ])

    table = Table(table_rows, colWidths=[45, 60, 45, 45, 40, 45, 50, 65, 85])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Policy Notice:</b> Recommendations prioritize SKUs exhibiting stock levels below calculated Reorder Points (ROP) during supplier lead times. Quantities are calculated based on Order-Up-To Targets: Q = max(0, Target - Inventory Position).", styles["note"]))

    doc.build(story)
    buf.seek(0)
    return buf


def generate_model_benchmark_pdf(metrics: Dict[str, Any]) -> io.BytesIO:
    """Generates College Model Evaluation & Academic Audit PDF Report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = create_base_styles()
    story = []

    add_header(story, "Academic Model Benchmark & Audit Report", "Chronological 80/20 Out-of-Sample Evaluation (14,700 Test Observations)", styles)

    story.append(Paragraph("1. Performance Comparison Matrix", styles["h2"]))
    table_rows = [
        [
            Paragraph("<b>Model Architecture</b>", styles["body"]),
            Paragraph("<b>MAE (Units)</b>", styles["body"]),
            Paragraph("<b>RMSE (Units)</b>", styles["body"]),
            Paragraph("<b>R² Score</b>", styles["body"]),
            Paragraph("<b>WAPE (%)</b>", styles["body"])
        ]
    ]

    for model_name, row in metrics.items():
        is_leak = "Leaked" in model_name or "Legacy" in model_name
        is_best = "Champion" in model_name or "Linear" in model_name
        table_rows.append([
            Paragraph(f"{'<b>' if is_best else ''}{model_name}{'</b>' if is_best else ''}", styles["body"]),
            Paragraph(f"{row.get('MAE', 0.0):.2f}", styles["body"]),
            Paragraph(f"{row.get('RMSE', 0.0):.2f}", styles["body"]),
            Paragraph(f"{row.get('R2', 0.0):.4f}", styles["body"]),
            Paragraph(f"{row.get('WAPE_pct', 0.0):.2f}%", styles["body"])
        ])

    table = Table(table_rows, colWidths=[200, 85, 85, 85, 85])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("2. Critical Data Leakage Audit Findings", styles["h2"]))
    story.append(Paragraph("• <b>Legacy Flaw:</b> In raw formulations, predicting 'Demand Forecast' using contemporaneous 'Units Sold' created an inverted task with 100% data leakage (R² = 0.9937). On the morning of a business day, store managers do not know contemporaneous units sold.<br/>"
                           "• <b>Honest Formulation:</b> StockSense AI designates actual demand (Units Sold) as the true prediction target, strictly excluding contemporaneous outcome variables.<br/>"
                           "• <b>Chronological Validation:</b> Standard random shuffling leaks future seasonality. StockSense AI enforces a strict chronological cutoff on August 8, 2023.", styles["body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Operational Trade-Off Analysis", styles["h2"]))
    story.append(Paragraph("Linear Regression provides transparent closed-form coefficients and slightly lower MAE (68.59 vs 68.70), while LightGBM Regressor effectively handles complex non-linear categorical interactions between store locations, promotions, and product categories.", styles["body"]))

    doc.build(story)
    buf.seek(0)
    return buf


def generate_category_analytics_pdf(cat_data: Dict[str, Any]) -> io.BytesIO:
    """Generates Category Demand Breakdown PDF Report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = create_base_styles()
    story = []

    add_header(story, "Retail Category Demand & Market Share Report", "Aggregated Demand Breakdown across 5 Core Product Categories", styles)

    cats = cat_data.get("categories", [])
    units = cat_data.get("total_units", [])
    shares = cat_data.get("market_share_pct", [])
    revs = cat_data.get("estimated_revenue", [])

    table_rows = [
        [
            Paragraph("<b>Category</b>", styles["body"]),
            Paragraph("<b>Total Units Sold</b>", styles["body"]),
            Paragraph("<b>Market Share</b>", styles["body"]),
            Paragraph("<b>Estimated Revenue</b>", styles["body"])
        ]
    ]

    for c, u, s, r in zip(cats, units, shares, revs):
        table_rows.append([
            Paragraph(f"<b>{c}</b>", styles["body"]),
            Paragraph(f"{u:,} Units", styles["body"]),
            Paragraph(f"{s}%", styles["body"]),
            Paragraph(format_inr(r, 2), styles["body"])
        ])

    table = Table(table_rows, colWidths=[140, 130, 130, 140])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    high = cat_data.get("highest_demand_category", {})
    low = cat_data.get("lowest_demand_category", {})
    story.append(Paragraph(f"• <b>Highest Volume Category:</b> {high.get('category')} ({high.get('units', 0):,} units sold)<br/>"
                           f"• <b>Lowest Volume Category:</b> {low.get('category')} ({low.get('units', 0):,} units sold)<br/>"
                           f"• <b>Total Catalog Demand:</b> {cat_data.get('total_catalog_units', 0):,} units across all stores.<br/>"
                           f"• <b>Currency Specification:</b> Category revenues are denominated in Indian Rupees (INR ₹) under the Indian numbering system.", styles["body"]))

    doc.build(story)
    buf.seek(0)
    return buf


def generate_academic_summary_pdf(all_metrics: Dict[str, Any], cat_data: Dict[str, Any]) -> io.BytesIO:
    """Generates Comprehensive Academic Project Submission Summary PDF."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = create_base_styles()
    story = []

    add_header(story, "StockSense AI: Academic Capstone Summary", "B.Tech Semester V — AIML Project Submission", styles)

    story.append(Paragraph("Executive Overview", styles["h2"]))
    story.append(Paragraph("StockSense AI integrates time-series machine learning regressors with continuous-review operations research policies (s, Q) to solve retail understocking and overstocking balance sheet risks.", styles["body"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Verified Model Benchmark Table", styles["h2"]))
    table_rows = [
        [
            Paragraph("<b>Model</b>", styles["body"]),
            Paragraph("<b>MAE</b>", styles["body"]),
            Paragraph("<b>RMSE</b>", styles["body"]),
            Paragraph("<b>R² Score</b>", styles["body"]),
            Paragraph("<b>WAPE (%)</b>", styles["body"])
        ]
    ]
    for m, r in all_metrics.items():
        table_rows.append([
            Paragraph(f"{m}", styles["body"]),
            Paragraph(f"{r.get('MAE', 0.0):.2f}", styles["body"]),
            Paragraph(f"{r.get('RMSE', 0.0):.2f}", styles["body"]),
            Paragraph(f"{r.get('R2', 0.0):.4f}", styles["body"]),
            Paragraph(f"{r.get('WAPE_pct', 0.0):.2f}%", styles["body"])
        ])
    t = Table(table_rows, colWidths=[180, 80, 80, 80, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("System Architecture & Inventory Formulas", styles["h2"]))
    story.append(Paragraph("• <b>Safety Stock:</b> SS = Z * σ_d * √L (Z=1.65 for 95% Cycle Service Level)<br/>"
                           "• <b>Reorder Point:</b> ROP = (d_hat * L) + SS<br/>"
                           "• <b>Stockout Risk:</b> P(Stockout) = 1 - Φ((Inventory Position - d_hat * L) / (σ_d * √L))<br/>"
                           "• <b>Target Order:</b> Q = max(0, Order-Up-To - Inventory Position)<br/>"
                           "• <b>Currency Specification:</b> All monetary values (prices, holding costs, revenues) are evaluated in Indian Rupees (INR ₹) under the Indian numbering system.", styles["body"]))

    doc.build(story)
    buf.seek(0)
    return buf
