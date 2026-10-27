/**
 * StockSense AI - Enterprise Frontend Controller
 * Complete implementation of 11 views, Chart.js visualizations,
 * interactive what-if simulations, CSV validation, persistent history,
 * and multi-SKU inventory optimization policies.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- State Variables ---
    let appMetadata = {};
    let activePaneId = "pane-summary";
    let isInitialized = false;

    // Chart.js instances
    let chartInstances = {};

    // --- Currency Formatters (Indian Rupee INR ₹ & Indian Numbering System) ---
    const inrCurrencyFormatter = new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });

    const inrCurrencyFormatterNoDecimals = new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    });

    function formatINR(val, includeDecimals = true) {
        if (val === null || val === undefined || isNaN(val)) {
            return includeDecimals ? "₹0.00" : "₹0";
        }
        const num = Number(val);
        return includeDecimals ? inrCurrencyFormatter.format(num) : inrCurrencyFormatterNoDecimals.format(num);
    }

    // --- DOM Cache ---
    const navItems = document.querySelectorAll(".nav-item");
    const contentPanes = document.querySelectorAll(".content-pane");
    const pageTitle = document.getElementById("pageTitle");
    const pageDesc = document.getElementById("pageDesc");
    const globalStoreSelect = document.getElementById("globalStoreSelect");
    const globalProductSelect = document.getElementById("globalProductSelect");
    const globalModelSelect = document.getElementById("globalModelSelect");
    const btnGlobalRefresh = document.getElementById("btnGlobalRefresh");
    const footerActiveModel = document.getElementById("footerActiveModel");
    const sidebarAlertCount = document.getElementById("sidebarAlertCount");

    // Toast
    const toastNotification = document.getElementById("toastNotification");
    const toastMessage = document.getElementById("toastMessage");

    function showToast(msg) {
        toastMessage.textContent = msg;
        toastNotification.classList.add("show");
        setTimeout(() => toastNotification.classList.remove("show"), 3500);
    }

    // --- Pane Meta Definitions ---
    const PANE_METAS = {
        "pane-summary": { title: "Forecast Summary", desc: "Operational SKU demand forecast, continuous-review safety buffer, and 7-day projection." },
        "pane-product": { title: "Product-Level Forecast", desc: "Granular ML feature scenario testing, 95% confidence intervals, and local attribution." },
        "pane-csv": { title: "CSV Validation & Batch", desc: "Drag-and-drop batch dataset validation, schema auditing, and chronological LightGBM batch prediction." },
        "pane-history": { title: "Forecast History", desc: "Persistent SQLite audit record of real forecast executions, filters, and CSV export." },
        "pane-inventory": { title: "Inventory Recommendation", desc: "Continuous review (s, Q) replenishment policy, safety stocks, and order sizing." },
        "pane-category": { title: "Demand by Category", desc: "Aggregated sales volume, estimated revenues, and market share across 5 core categories." },
        "pane-seasonality": { title: "Sales & Seasonality Analysis", desc: "Empirical sales trends by month, day of week, seasonal cycles, and promotional uplift." },
        "pane-pricing": { title: "Price Impact & What-If Simulator", desc: "Price elasticity scatter, discount tiers, and real-time model scenario simulation." },
        "pane-alerts": { title: "Stockout Risk Alerts", desc: "Ranked monitoring of multi-store SKU stockout vulnerabilities and automated policy triggers." },
        "pane-benchmark": { title: "Model Benchmark & XAI", desc: "Chronological 80/20 test set evaluation matrix, data leakage audit, and feature importances." },
        "pane-reports": { title: "Reports & Export Hub", desc: "Publication-quality PDF downloads and raw CSV data exports." }
    };

    // --- Navigation Controller ---
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const paneId = item.getAttribute("data-pane");
            switchPane(paneId);
        });
    });

    function switchPane(paneId) {
        activePaneId = paneId;
        navItems.forEach(n => n.classList.remove("active"));
        contentPanes.forEach(p => p.classList.remove("active"));

        const targetNav = document.querySelector(`.nav-item[data-pane="${paneId}"]`);
        const targetPane = document.getElementById(paneId);

        if (targetNav) targetNav.classList.add("active");
        if (targetPane) targetPane.classList.add("active");

        const meta = PANE_METAS[paneId] || { title: "Analytics", desc: "" };
        pageTitle.textContent = meta.title;
        pageDesc.textContent = meta.desc;

        // Trigger view-specific loads
        loadViewData(paneId);
    }

    function destroyChart(key) {
        if (chartInstances[key]) {
            chartInstances[key].destroy();
            chartInstances[key] = null;
        }
    }

    // --- 1. Global Initialization ---
    async function initApp() {
        try {
            const resp = await fetch("/api/metadata");
            const data = await resp.json();
            appMetadata = data;

            // Populate global product dropdown
            globalProductSelect.innerHTML = "";
            const cat = data.sku_catalogue || {};
            (data.products || []).forEach(pId => {
                const opt = document.createElement("option");
                opt.value = pId;
                const info = cat[pId] || {};
                opt.textContent = `${pId} — ${info.name || pId} (${info.category || ''})`;
                globalProductSelect.appendChild(opt);
            });

            // Populate product-level SKU dropdown
            const prodProductSelect = document.getElementById("prodProductSelect");
            if (prodProductSelect) {
                prodProductSelect.innerHTML = globalProductSelect.innerHTML;
            }

            // Sync model footer
            updateModelFooter();

            // Load initial view
            isInitialized = true;
            loadViewData("pane-summary");
            updateSidebarAlertBadge();

            // Populate authenticated user info in topbar
            try {
                const authResp = await fetch("/api/auth/me");
                const authData = await authResp.json();
                if (authData.authenticated && authData.user) {
                    const u = authData.user;
                    const nameLabel = document.getElementById("userNameLabel");
                    const roleLabel = document.getElementById("userRoleLabel");
                    const avatarInitials = document.getElementById("userAvatarInitials");
                    if (nameLabel) nameLabel.textContent = u.full_name || "Enterprise User";
                    if (roleLabel) roleLabel.textContent = u.role || "Analyst";
                    if (avatarInitials && u.full_name) {
                        const parts = u.full_name.trim().split(/\s+/);
                        const initials = parts.length > 1 ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase() : parts[0].substring(0, 2).toUpperCase();
                        avatarInitials.textContent = initials;
                    }
                }
            } catch (err) {
                console.warn("Could not load user profile:", err);
            }
        } catch (e) {
            console.error("Initialization error:", e);
        }
    }

    function updateModelFooter() {
        const sel = globalModelSelect.options[globalModelSelect.selectedIndex].text;
        footerActiveModel.textContent = sel;
    }

    // Global selector events
    globalStoreSelect.addEventListener("change", () => loadViewData(activePaneId));
    globalProductSelect.addEventListener("change", () => loadViewData(activePaneId));
    globalModelSelect.addEventListener("change", () => {
        updateModelFooter();
        loadViewData(activePaneId);
    });
    btnGlobalRefresh.addEventListener("click", () => {
        showToast("Refreshing data from server...");
        loadViewData(activePaneId);
        updateSidebarAlertBadge();
    });

    // View Router Dispatcher
    function loadViewData(paneId) {
        const storeId = globalStoreSelect.value;
        const productId = globalProductSelect.value;
        const modelChoice = globalModelSelect.value;

        switch (paneId) {
            case "pane-summary":
                loadSummaryView(storeId, productId, modelChoice);
                break;
            case "pane-product":
                syncProductViewInputs(storeId, productId);
                break;
            case "pane-history":
                loadHistoryTable();
                break;
            case "pane-inventory":
                loadInventoryMatrix(storeId, modelChoice);
                break;
            case "pane-category":
                loadCategoryDemand();
                break;
            case "pane-seasonality":
                loadSeasonalityTrends(storeId);
                break;
            case "pane-pricing":
                loadPricingAnalytics();
                break;
            case "pane-alerts":
                loadStockoutAlerts();
                break;
            case "pane-benchmark":
                loadBenchmarkMatrix();
                loadExplainabilityCharts();
                break;
        }
    }

    // ==============================================================
    // PANE 1: FORECAST SUMMARY (Figure 4.5.1)
    // ==============================================================
    async function loadSummaryView(storeId, productId, modelChoice) {
        try {
            // 1. Fetch prediction & inventory evaluation
            const predResp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ store_id: storeId, product_id: productId, model_choice: modelChoice })
            });
            const predJson = await predResp.json();
            if (!predJson.success) return;
            const res = predJson.data;

            // 2. Fetch 30-day history + 7-day projection
            const histResp = await fetch(`/api/history?store_id=${storeId}&product_id=${productId}&model_choice=${modelChoice}`);
            const histJson = await histResp.json();
            if (!histJson.success) return;

            // Update sub-header tag
            document.getElementById("summarySkuTag").textContent = `${res.product_name} (${storeId} - ${res.region} Region)`;

            // Update 4 cards
            const fc = res.forecast;
            const inv = res.inventory;

            document.getElementById("summaryPredictedDemand").textContent = fc.predicted_units_sold;
            document.getElementById("summaryDemandCI").textContent = `95% CI: [${fc.confidence_interval_95.lower} - ${fc.confidence_interval_95.upper}]`;

            document.getElementById("summaryCurrentInventory").textContent = inv.current_inventory;
            document.getElementById("summaryDaysSupply").textContent = `${inv.days_of_supply} Days Supply`;

            document.getElementById("summaryReorderPoint").textContent = inv.reorder_point;
            document.getElementById("summarySafetyStockBadge").textContent = `SS: ${inv.safety_stock} Units`;

            document.getElementById("summaryStockoutRisk").textContent = `${inv.stockout_risk_pct}%`;
            document.getElementById("summaryRiskTier").textContent = inv.stockout_risk_pct > 25 ? "Critical" : (inv.stockout_risk_pct > 10 ? "Elevated" : "Low");

            // Update Replenishment Decision Banner
            const bannerBox = document.getElementById("summaryActionStatusBox");
            const headline = document.getElementById("summaryOrderHeadline");
            const safetySub = document.getElementById("summarySafetySub");

            if (inv.recommended_order_qty > 0) {
                bannerBox.textContent = "REORDER RECOMMENDED";
                bannerBox.className = "action-status-box";
                headline.innerHTML = `Order <strong>${inv.recommended_order_qty} Units</strong>`;
                safetySub.innerHTML = `With safety stock buffer <strong>${inv.safety_stock} Units</strong> | Cycle Service Level 95% (Z=1.65)`;
            } else {
                bannerBox.textContent = "STOCK OPTIMAL";
                bannerBox.className = "action-status-box optimal";
                headline.innerHTML = `Inventory Position <strong>${inv.inventory_position} Units</strong> Sufficient`;
                safetySub.innerHTML = `Days of supply: <strong>${inv.days_of_supply} days</strong> | Safety buffer maintained at <strong>${inv.safety_stock} units</strong>`;
            }

            // Render mini sparklines
            renderSparkline("sparklineDemand", [80, 95, 110, 105, 130, 140, fc.predicted_units_sold], "#06b6d4");
            renderSparkline("sparklineInventory", [inv.current_inventory + 20, inv.current_inventory + 10, inv.current_inventory, inv.current_inventory], "#10b981");
            renderSparkline("sparklineROP", [inv.reorder_point, inv.reorder_point, inv.reorder_point], "#f59e0b");
            renderSparkline("sparklineRisk", [28, 22, 19, inv.stockout_risk_pct], "#f43f5e");

            // Render Main Dual-Axis Timeseries Chart
            renderMainTimeseries(histJson);

        } catch (e) {
            console.error("Summary view load error:", e);
        }
    }

    function renderSparkline(canvasId, dataPoints, color) {
        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        chartInstances[canvasId] = new Chart(ctx, {
            type: "line",
            data: {
                labels: dataPoints.map((_, i) => i),
                datasets: [{
                    data: dataPoints,
                    borderColor: color,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false } }
            }
        });
    }

    function renderMainTimeseries(data) {
        destroyChart("mainTimeseries");
        const ctx = document.getElementById("mainTimeseriesChart");
        if (!ctx) return;

        const histDates = data.historical.dates;
        const histSales = data.historical.units_sold;
        const histInv = data.historical.inventory_level;

        const fcDates = data.forecast.dates;
        const fcSales = data.forecast.units_sold;
        const ciLower = data.forecast.ci_lower;
        const ciUpper = data.forecast.ci_upper;

        const allDates = [...histDates, ...fcDates];
        const paddedHistSales = [...histSales, ...Array(fcDates.length).fill(null)];
        const paddedHistInv = [...histInv, ...Array(fcDates.length).fill(null)];
        const paddedFcSales = [...Array(histDates.length - 1).fill(null), histSales[histSales.length - 1], ...fcSales];

        chartInstances["mainTimeseries"] = new Chart(ctx, {
            type: "line",
            data: {
                labels: allDates,
                datasets: [
                    {
                        label: "Actual Units Sold",
                        data: paddedHistSales,
                        borderColor: "#06b6d4",
                        backgroundColor: "rgba(6, 182, 212, 0.15)",
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: "Historical Inventory Level",
                        data: paddedHistInv,
                        borderColor: "#10b981",
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        tension: 0.2
                    },
                    {
                        label: "7-Day ML Forecast",
                        data: paddedFcSales,
                        borderColor: "#a855f7",
                        backgroundColor: "rgba(168, 85, 247, 0.18)",
                        borderWidth: 2.5,
                        pointRadius: 4,
                        pointBackgroundColor: "#c084fc",
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans", size: 12 } } },
                    tooltip: {
                        backgroundColor: "rgba(15, 23, 42, 0.95)",
                        titleColor: "#f8fafc",
                        bodyColor: "#94a3b8",
                        borderColor: "rgba(255, 255, 255, 0.1)",
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#64748b", maxTicksLimit: 12, font: { family: "JetBrains Mono", size: 10 } }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 11 } }
                    }
                }
            }
        });
    }

    // PDF Summary button
    document.getElementById("btnDownloadSummaryPdf").addEventListener("click", () => {
        const s = globalStoreSelect.value;
        const p = globalProductSelect.value;
        const m = globalModelSelect.value;
        window.location.href = `/api/export/pdf/forecast?store_id=${s}&product_id=${p}&model_choice=${m}`;
    });

    // ==============================================================
    // PANE 2: PRODUCT-LEVEL FORECAST (Figure 4.5.2)
    // ==============================================================
    function syncProductViewInputs(storeId, productId) {
        document.getElementById("prodStoreSelect").value = storeId;
        const pSel = document.getElementById("prodProductSelect");
        if (pSel) pSel.value = productId;
        document.getElementById("prodDateInput").valueAsDate = new Date();
        executeProductLevelForecast();
    }

    document.getElementById("productForecastForm").addEventListener("submit", (e) => {
        e.preventDefault();
        executeProductLevelForecast();
    });

    async function executeProductLevelForecast() {
        const sId = document.getElementById("prodStoreSelect").value;
        const pId = document.getElementById("prodProductSelect").value;
        const dt = document.getElementById("prodDateInput").value;
        const horizon = document.getElementById("prodHorizonInput").value;
        const price = document.getElementById("prodPriceInput").value;
        const discount = document.getElementById("prodDiscountInput").value;
        const compPrice = document.getElementById("prodCompPriceInput").value;
        const weather = document.getElementById("prodWeatherSelect").value;
        const holiday = document.getElementById("prodHolidaySelect").value;
        const stock = document.getElementById("prodInventoryInput").value;
        const incoming = document.getElementById("prodIncomingInput").value;
        const leadTime = document.getElementById("prodLeadTimeInput").value;
        const modelChoice = globalModelSelect.value;

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    store_id: sId,
                    product_id: pId,
                    date: dt,
                    price: price,
                    discount: discount,
                    competitor_pricing: compPrice,
                    weather_condition: weather,
                    holiday_promotion: holiday,
                    current_inventory: stock,
                    incoming_orders: incoming,
                    lead_time_days: leadTime,
                    forecast_horizon_days: horizon,
                    model_choice: modelChoice
                })
            });
            const json = await resp.json();
            if (!json.success) return;
            const res = json.data;

            // Update UI Outputs
            document.getElementById("prodModelBadge").textContent = res.model_used;
            document.getElementById("prodOutputUnits").textContent = res.forecast.predicted_units_sold;
            document.getElementById("prodOutputCI").textContent = `95% CI: [${res.forecast.confidence_interval_95.lower} – ${res.forecast.confidence_interval_95.upper}] Units`;
            document.getElementById("prodOutputRevenue").textContent = `Estimated Daily Revenue: ${formatINR(res.forecast.expected_revenue, true)}`;

            // Render Attribution Bars
            const container = document.getElementById("prodAttributionBars");
            container.innerHTML = "";
            (res.feature_contributions || []).forEach(f => {
                const row = document.createElement("div");
                row.className = "att-row";
                const isPos = f.impact_units >= 0;
                const fillPct = Math.min(100, Math.max(10, Math.abs(f.impact_units) * 2.5));

                row.innerHTML = `
                    <span class="att-label">${f.feature} (${f.value})</span>
                    <div class="att-bar-track">
                        <div class="att-bar-fill" style="width: ${fillPct}%; background: ${isPos ? 'linear-gradient(90deg, #10b981, #34d399)' : 'linear-gradient(90deg, #f43f5e, #fb7185)'}"></div>
                    </div>
                    <span class="att-val" style="color: ${isPos ? '#34d399' : '#fb7185'}">${isPos ? '+' : ''}${f.impact_units} U</span>
                `;
                container.appendChild(row);
            });

            // Quick Inventory Metrics
            const inv = res.inventory;
            document.getElementById("prodQuickSS").textContent = `${inv.safety_stock} Units`;
            document.getElementById("prodQuickROP").textContent = `${inv.reorder_point} Units`;
            document.getElementById("prodQuickRisk").textContent = `${inv.stockout_risk_pct}%`;
            document.getElementById("prodQuickOrder").textContent = `${inv.recommended_order_qty} Units`;

            showToast("Forecast generated and saved to history!");
            updateSidebarAlertBadge();

        } catch (e) {
            console.error("Product forecast error:", e);
        }
    }

    // ==============================================================
    // PANE 3: CSV VALIDATION & BATCH FORECAST (Figure 4.5.3)
    // ==============================================================
    const csvDropZone = document.getElementById("csvDropZone");
    const csvFileInput = document.getElementById("csvFileInput");

    if (csvDropZone) {
        csvDropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            csvDropZone.classList.add("dragover");
        });
        csvDropZone.addEventListener("dragleave", () => csvDropZone.classList.remove("dragover"));
        csvDropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            csvDropZone.classList.remove("dragover");
            if (e.dataTransfer.files.length) handleCSVUpload(e.dataTransfer.files[0]);
        });
    }

    if (csvFileInput) {
        csvFileInput.addEventListener("change", (e) => {
            if (e.target.files.length) handleCSVUpload(e.target.files[0]);
        });
    }

    async function handleCSVUpload(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/api/upload_csv", { method: "POST", body: formData });
            const json = await resp.json();
            if (!json.success) {
                alert(json.error);
                return;
            }

            document.getElementById("dropFileName").textContent = json.filename;
            document.getElementById("csvSchemaBadge").textContent = json.is_valid ? "Schema Validated (0 Nulls, 100% Type Match)" : "Schema Invalid";
            document.getElementById("csvSchemaBadge").className = json.is_valid ? "badge-status-pill pill-model" : "badge-status-pill";

            document.getElementById("statRowCount").textContent = `${json.row_count.toLocaleString()} Records`;
            document.getElementById("statColCount").textContent = `${json.col_count} Columns`;
            document.getElementById("statNullCount").textContent = `${json.total_nulls} Nulls`;
            document.getElementById("statDupCount").textContent = `${json.duplicate_rows} Duplicates`;
            document.getElementById("csvFileStats").style.display = "flex";

            // Populate preview
            const tb = document.getElementById("csvPreviewTableBody");
            tb.innerHTML = "";
            (json.preview || []).forEach(r => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${r.Date || '-'}</td>
                    <td>${r['Store ID'] || '-'}</td>
                    <td>${r['Product ID'] || '-'}</td>
                    <td>${formatINR(parseFloat(r.Price || 0), true)}</td>
                    <td>${r['Inventory Level'] || 0}</td>
                    <td>${r.Discount || 0}%</td>
                `;
                tb.appendChild(tr);
            });

            showToast("CSV file successfully audited and validated!");

        } catch (e) {
            console.error("CSV upload error:", e);
        }
    }

    document.getElementById("btnExecuteBatchForecast").addEventListener("click", async () => {
        const fill = document.getElementById("batchProgressFill");
        const pLabel = document.getElementById("batchProgressLabel");
        const pText = document.getElementById("batchPercentText");

        pLabel.textContent = "Processing chronological pipeline batch...";
        fill.style.width = "45%";
        pText.textContent = "45%";

        try {
            const resp = await fetch("/api/batch_forecast", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_choice: globalModelSelect.value, max_rows: 250 })
            });
            const json = await resp.json();
            if (!json.success) return;

            fill.style.width = "100%";
            pText.textContent = "100%";
            pLabel.textContent = "Batch predictions complete!";

            document.getElementById("batchCountDisplay").textContent = `${json.processed_records} Predictions Generated`;
            document.getElementById("batchMAE").textContent = `${json.mean_absolute_error} Units`;
            document.getElementById("batchTime").textContent = `${json.processing_time_sec}s`;

            showToast(`Batch forecast generated for ${json.processed_records} rows!`);

        } catch (e) {
            console.error("Batch forecast error:", e);
        }
    });

    // ==============================================================
    // PANE 4: FORECAST HISTORY (Figure 4.5.4)
    // ==============================================================
    async function loadHistoryTable() {
        const storeFilter = document.getElementById("historyStoreFilter").value;
        const search = document.getElementById("historySearchInput").value;

        try {
            const resp = await fetch(`/api/forecast_history?store_id=${storeFilter}&search=${encodeURIComponent(search)}&limit=100`);
            const json = await resp.json();
            if (!json.success) return;
            const data = json.data;

            document.getElementById("historyRecordCount").textContent = `${data.total} Runs`;
            document.getElementById("historyServingModel").textContent = globalModelSelect.options[globalModelSelect.selectedIndex].text;

            const tb = document.getElementById("historyTableBody");
            tb.innerHTML = "";

            if (!data.runs.length) {
                tb.innerHTML = `<tr><td colspan="12" class="text-center text-muted">No forecast runs recorded yet. Generate forecasts to log persistent audits.</td></tr>`;
                return;
            }

            data.runs.forEach(r => {
                const tr = document.createElement("tr");
                const isRisk = r.stockout_risk_pct > 20;
                tr.innerHTML = `
                    <td><span class="badge-sub font-mono">${r.run_id}</span></td>
                    <td>${r.created_at}</td>
                    <td><strong>${r.store_id}</strong> / ${r.product_id}</td>
                    <td>${r.category}</td>
                    <td>${r.model_used.split(' ')[0]}</td>
                    <td><strong>${r.predicted_units}</strong></td>
                    <td>${r.inventory_position}</td>
                    <td>${r.reorder_point}</td>
                    <td><span style="color: ${isRisk ? '#f43f5e' : '#10b981'}">${r.stockout_risk_pct}%</span></td>
                    <td><strong>${r.recommended_order}</strong></td>
                    <td><span class="badge-sub">${r.status_label}</span></td>
                    <td>
                        <button class="btn-danger-outline btn-sm btn-delete-run" data-id="${r.run_id}">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </td>
                `;
                tb.appendChild(tr);
            });

            // Bind delete buttons
            document.querySelectorAll(".btn-delete-run").forEach(btn => {
                btn.addEventListener("click", async () => {
                    const runId = btn.getAttribute("data-id");
                    if (confirm(`Delete audit run ${runId}?`)) {
                        await fetch(`/api/forecast_history/${runId}`, { method: "DELETE" });
                        showToast(`Run ${runId} deleted.`);
                        loadHistoryTable();
                    }
                });
            });

        } catch (e) {
            console.error("History table error:", e);
        }
    }

    document.getElementById("historyStoreFilter").addEventListener("change", loadHistoryTable);
    document.getElementById("historySearchInput").addEventListener("input", debounce(loadHistoryTable, 300));
    document.getElementById("btnClearHistory").addEventListener("click", async () => {
        if (confirm("Clear all persistent forecast history records?")) {
            await fetch("/api/forecast_history/clear", { method: "POST" });
            showToast("Forecast history cleared.");
            loadHistoryTable();
        }
    });

    // ==============================================================
    // PANE 5: INVENTORY RECOMMENDATION (Figure 4.5.5)
    // ==============================================================
    async function loadInventoryMatrix(storeId, modelChoice) {
        try {
            const resp = await fetch(`/api/inventory_matrix?store_id=${storeId}&model_choice=${modelChoice}`);
            const json = await resp.json();
            if (!json.success) return;

            const tb = document.getElementById("storeMatrixTableBody");
            tb.innerHTML = "";

            let totalCost = 0;
            let criticalCount = 0;

            json.matrix.forEach(item => {
                const tr = document.createElement("tr");
                const isDanger = item.severity === "danger";
                const isWarning = item.severity === "warning";
                if (isDanger) criticalCount++;

                const orderCost = item.recommended_order * item.price;
                totalCost += orderCost;

                const statusColor = isDanger ? "#f43f5e" : (isWarning ? "#f59e0b" : "#10b981");

                tr.innerHTML = `
                    <td><strong>${item.product_id}</strong></td>
                    <td>${item.product_name}</td>
                    <td>${item.category}</td>
                    <td>${formatINR(item.price, true)}</td>
                    <td>${item.current_inventory}</td>
                    <td><strong>${item.predicted_demand}</strong></td>
                    <td>${item.safety_stock}</td>
                    <td>${item.reorder_point}</td>
                    <td style="color: ${statusColor}; font-weight: 700;">${item.stockout_risk_pct}%</td>
                    <td><strong>${item.recommended_order}</strong></td>
                    <td><span class="badge-sub" style="color: ${statusColor}">${item.status_label}</span></td>
                `;
                tb.appendChild(tr);
            });

            // Update bottom bar
            document.getElementById("invEstimatedCost").textContent = formatINR(totalCost, true);
            document.getElementById("invHeaderPriorityText").textContent = criticalCount > 0 ? `REORDER REQUIRED - ${criticalCount} Critical SKUs` : "OPTIMAL REPLENISHMENT POSITION";

        } catch (e) {
            console.error("Inventory matrix error:", e);
        }
    }

    // ==============================================================
    // PANE 6: DEMAND BY CATEGORY (Figure 4.5.6)
    // ==============================================================
    async function loadCategoryDemand() {
        try {
            const resp = await fetch("/api/analytics/category_demand");
            const json = await resp.json();
            if (!json.success) return;
            const data = json.data;

            document.getElementById("catHighName").textContent = data.highest_demand_category.category;
            document.getElementById("catHighUnits").textContent = `${data.highest_demand_category.units.toLocaleString()} Units (${formatINR(data.highest_demand_category.revenue, false)})`;

            document.getElementById("catLowName").textContent = data.lowest_demand_category.category;
            document.getElementById("catLowUnits").textContent = `${data.lowest_demand_category.units.toLocaleString()} Units (${formatINR(data.lowest_demand_category.revenue, false)})`;

            document.getElementById("catTotalVolume").textContent = `${data.total_catalog_units.toLocaleString()} Units`;
            document.getElementById("catTotalUnitsPill").textContent = `${data.total_catalog_units.toLocaleString()} Test Units`;

            // Horizontal Bar Chart
            renderCategoryBarChart(data);

            // Donut Chart
            renderCategoryDonutChart(data);

        } catch (e) {
            console.error("Category demand error:", e);
        }
    }

    function renderCategoryBarChart(data) {
        destroyChart("catBar");
        const ctx = document.getElementById("categoryBarChart");
        if (!ctx) return;

        chartInstances["catBar"] = new Chart(ctx, {
            type: "bar",
            data: {
                labels: data.categories,
                datasets: [{
                    label: "Predicted Units Sold",
                    data: data.total_units,
                    backgroundColor: ["#10b981", "#06b6d4", "#3b82f6", "#f59e0b", "#a855f7"],
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Sales: ${ctx.raw.toLocaleString()} Units (${formatINR(data.estimated_revenue[ctx.dataIndex], false)} Revenue)`
                        }
                    }
                },
                scales: {
                    x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#64748b" } },
                    y: { grid: { display: false }, ticks: { color: "#f8fafc", font: { weight: 600 } } }
                }
            }
        });
    }

    function renderCategoryDonutChart(data) {
        destroyChart("catDonut");
        const ctx = document.getElementById("categoryDonutChart");
        if (!ctx) return;

        chartInstances["catDonut"] = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: data.categories,
                datasets: [{
                    data: data.market_share_pct,
                    backgroundColor: ["#10b981", "#06b6d4", "#3b82f6", "#f59e0b", "#a855f7"],
                    borderWidth: 2,
                    borderColor: "#0b1329"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "68%",
                plugins: {
                    legend: { position: "right", labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans" } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.label}: ${ctx.raw}% share`
                        }
                    }
                }
            }
        });
    }

    // ==============================================================
    // PANE 7: SALES TREND & SEASONALITY (Feature 7)
    // ==============================================================
    async function loadSeasonalityTrends(storeId) {
        try {
            const resp = await fetch(`/api/analytics/sales_trends?store_id=${storeId}`);
            const json = await resp.json();
            if (!json.success) return;
            const data = json.data;

            document.getElementById("seasonTotalSales").textContent = data.summary.total_units_sold.toLocaleString();
            document.getElementById("seasonAvgDaily").textContent = data.summary.average_daily_demand;
            document.getElementById("seasonPeakDate").textContent = data.summary.peak_demand_date;
            document.getElementById("seasonPeakUnits").textContent = `Volume: ${data.summary.peak_daily_units.toLocaleString()} Units`;
            document.getElementById("seasonPromoUplift").textContent = `+${data.summary.promo_uplift_pct}%`;

            // Monthly Trend Line
            destroyChart("monthlyTrend");
            const mCtx = document.getElementById("monthlyTrendChart");
            if (mCtx) {
                chartInstances["monthlyTrend"] = new Chart(mCtx, {
                    type: "line",
                    data: {
                        labels: data.monthly_trend.labels,
                        datasets: [{
                            label: "Total Units Sold",
                            data: data.monthly_trend.total_sales,
                            borderColor: "#06b6d4",
                            backgroundColor: "rgba(6, 182, 212, 0.12)",
                            fill: true,
                            tension: 0.35,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", maxTicksLimit: 12 } },
                            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } }
                        }
                    }
                });
            }

            // DOW Bar Chart
            destroyChart("dowChart");
            const dCtx = document.getElementById("dowChart");
            if (dCtx) {
                chartInstances["dowChart"] = new Chart(dCtx, {
                    type: "bar",
                    data: {
                        labels: data.day_of_week.labels,
                        datasets: [{
                            label: "Average Daily Demand",
                            data: data.day_of_week.average_sales,
                            backgroundColor: "#10b981",
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { display: false }, ticks: { color: "#f8fafc" } },
                            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } }
                        }
                    }
                });
            }

            // Seasonality Bar Chart
            destroyChart("seasonChart");
            const sCtx = document.getElementById("seasonChart");
            if (sCtx) {
                chartInstances["seasonChart"] = new Chart(sCtx, {
                    type: "bar",
                    data: {
                        labels: data.seasonality.labels,
                        datasets: [{
                            label: "Average Sales per SKU",
                            data: data.seasonality.average_sales,
                            backgroundColor: ["#34d399", "#f59e0b", "#fb923c", "#60a5fa"],
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { display: false }, ticks: { color: "#f8fafc" } },
                            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } }
                        }
                    }
                });
            }

        } catch (e) {
            console.error("Seasonality trends error:", e);
        }
    }

    // ==============================================================
    // PANE 8: PRICING ANALYTICS & WHAT-IF SIMULATOR (Feature 8)
    // ==============================================================
    async function loadPricingAnalytics() {
        try {
            const resp = await fetch("/api/analytics/price_promotion");
            const json = await resp.json();
            if (!json.success) return;
            const data = json.data;

            // Price Scatter
            destroyChart("priceScatter");
            const pCtx = document.getElementById("priceScatterChart");
            if (pCtx) {
                chartInstances["priceScatter"] = new Chart(pCtx, {
                    type: "scatter",
                    data: {
                        datasets: [{
                            label: "Price vs. Demand",
                            data: data.scatter_points.map(p => ({ x: p.x, y: p.y })),
                            backgroundColor: "rgba(6, 182, 212, 0.6)",
                            pointRadius: 4,
                            pointHoverRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: (ctx) => `Price: ${formatINR(ctx.raw.x, true)} | Demand: ${ctx.raw.y} Units`
                                }
                            }
                        },
                        scales: {
                            x: { title: { display: true, text: "Unit Price (₹)", color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } },
                            y: { title: { display: true, text: "Units Sold", color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } }
                        }
                    }
                });
            }

            // Discount Tier Bar
            destroyChart("discountBar");
            const dCtx = document.getElementById("discountBarChart");
            if (dCtx) {
                chartInstances["discountBar"] = new Chart(dCtx, {
                    type: "bar",
                    data: {
                        labels: data.discount_impact.labels,
                        datasets: [{
                            label: "Average Daily Demand",
                            data: data.discount_impact.average_sales,
                            backgroundColor: "#10b981",
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { display: false }, ticks: { color: "#f8fafc" } },
                            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } }
                        }
                    }
                });
            }

            // Initial simulation
            executeSimulation();

        } catch (e) {
            console.error("Pricing analytics error:", e);
        }
    }

    // Sliders
    const simPriceSlider = document.getElementById("simPriceSlider");
    const simDiscountSlider = document.getElementById("simDiscountSlider");
    const simCompSlider = document.getElementById("simCompSlider");
    const simInventorySlider = document.getElementById("simInventorySlider");

    [simPriceSlider, simDiscountSlider, simCompSlider, simInventorySlider].forEach(slider => {
        if (slider) slider.addEventListener("input", () => {
            document.getElementById("simPriceVal").textContent = formatINR(simPriceSlider.value, true);
            document.getElementById("simDiscountVal").textContent = `${simDiscountSlider.value}%`;
            document.getElementById("simCompPriceVal").textContent = formatINR(simCompSlider.value, true);
            document.getElementById("simInventoryVal").textContent = `${simInventorySlider.value} Units`;
            debounceSimulation();
        });
    });

    const debounceSimulation = debounce(executeSimulation, 250);

    async function executeSimulation() {
        const storeId = globalStoreSelect.value;
        const productId = globalProductSelect.value;
        const price = simPriceSlider ? simPriceSlider.value : 45;
        const discount = simDiscountSlider ? simDiscountSlider.value : 15;
        const comp = simCompSlider ? simCompSlider.value : 48;
        const inv = simInventorySlider ? simInventorySlider.value : 180;
        const model = globalModelSelect.value;

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    store_id: storeId,
                    product_id: productId,
                    price: price,
                    discount: discount,
                    competitor_pricing: comp,
                    current_inventory: inv,
                    model_choice: model
                })
            });
            const json = await resp.json();
            if (!json.success) return;
            const res = json.data;

            document.getElementById("simResultDemand").textContent = `${res.forecast.predicted_units_sold} Units`;
            document.getElementById("simResultRevenue").textContent = formatINR(res.forecast.expected_revenue, true);
            document.getElementById("simResultRisk").textContent = `${res.inventory.stockout_risk_pct}%`;
            document.getElementById("simResultSupply").textContent = `${res.inventory.days_of_supply} Days`;

        } catch (e) {
            console.error("Simulation error:", e);
        }
    }

    // ==============================================================
    // PANE 9: STOCKOUT ALERTS (Feature 9)
    // ==============================================================
    async function loadStockoutAlerts() {
        const sevFilter = document.getElementById("alertSeverityFilter").value;
        const search = document.getElementById("alertSearchInput").value;

        try {
            const resp = await fetch(`/api/analytics/stockout_alerts?severity=${sevFilter}`);
            const json = await resp.json();
            if (!json.success) return;
            const data = json.data;

            document.getElementById("alertCriticalCount").textContent = data.summary_counts.critical;
            document.getElementById("alertWarningCount").textContent = data.summary_counts.reorder_needed;
            document.getElementById("alertOptimalCount").textContent = data.summary_counts.optimal;
            document.getElementById("alertOverstockCount").textContent = data.summary_counts.overstocked;

            const tb = document.getElementById("alertsTableBody");
            tb.innerHTML = "";

            let filtered = data.alerts;
            if (search) {
                const s = search.toLowerCase();
                filtered = filtered.filter(a => a.product_id.toLowerCase().includes(s) || a.product_name.toLowerCase().includes(s) || a.store_id.toLowerCase().includes(s));
            }

            filtered.forEach((a, idx) => {
                const tr = document.createElement("tr");
                const sevColor = a.severity === "danger" ? "#f43f5e" : (a.severity === "warning" ? "#f59e0b" : "#10b981");

                tr.innerHTML = `
                    <td><strong>#${idx + 1}</strong> (Urgency ${a.urgency_score})</td>
                    <td><strong>${a.store_id}</strong></td>
                    <td>${a.product_id} — ${a.product_name}</td>
                    <td>${a.category}</td>
                    <td>${a.current_inventory}</td>
                    <td>${a.reorder_point}</td>
                    <td style="color: ${sevColor}; font-weight: 700;">${a.stockout_risk_pct}%</td>
                    <td><strong>${a.recommended_order_qty}</strong></td>
                    <td><span class="badge-sub" style="color: ${sevColor}">${a.status_label}</span></td>
                    <td style="font-size: 11.5px; color: #94a3b8; max-width: 280px; white-space: normal;">${a.reason}</td>
                `;
                tb.appendChild(tr);
            });

        } catch (e) {
            console.error("Stockout alerts error:", e);
        }
    }

    document.getElementById("alertSeverityFilter").addEventListener("change", loadStockoutAlerts);
    document.getElementById("alertSearchInput").addEventListener("input", debounce(loadStockoutAlerts, 300));

    async function updateSidebarAlertBadge() {
        try {
            const resp = await fetch("/api/analytics/stockout_alerts");
            const json = await resp.json();
            if (json.success) {
                const count = json.data.summary_counts.critical + json.data.summary_counts.reorder_needed;
                sidebarAlertCount.textContent = count;
            }
        } catch (e) {}
    }

    // ==============================================================
    // PANE 10: BENCHMARK MATRIX & XAI (Feature 11 + Phase 2)
    // ==============================================================
    async function loadBenchmarkMatrix() {
        try {
            const resp = await fetch("/api/benchmark_metrics");
            const json = await resp.json();
            if (!json.success) return;

            const tb = document.getElementById("benchmarkTableBody");
            tb.innerHTML = "";

            Object.entries(json.metrics).forEach(([modelName, row]) => {
                const tr = document.createElement("tr");
                const isChampion = modelName.includes("Champion");
                const isBaseline = modelName.includes("Baseline");
                const isLeak = modelName.includes("Leaked") || modelName.includes("Legacy");

                tr.innerHTML = `
                    <td><strong>${modelName}</strong></td>
                    <td><code>${isLeak ? 'Demand Forecast' : 'Units Sold'}</code></td>
                    <td>${row.MAE.toFixed(2)}</td>
                    <td>${row.RMSE.toFixed(2)}</td>
                    <td style="font-weight: 700; color: ${isLeak ? '#f43f5e' : (row.R2 > 0 ? '#10b981' : '#64748b')}">${row.R2.toFixed(4)}</td>
                    <td>${row.WAPE_pct.toFixed(2)}%</td>
                    <td><span class="badge-sub" style="color: ${isLeak ? '#f43f5e' : '#10b981'}">${isLeak ? '100% Leaked' : 'Zero Leakage'}</span></td>
                    <td>${isChampion ? 'Production Engine' : (isBaseline ? 'Student Baseline' : (isLeak ? 'Inverted Task' : 'Evaluation Baseline'))}</td>
                `;
                tb.appendChild(tr);
            });

        } catch (e) {
            console.error("Benchmark matrix error:", e);
        }
    }

    async function loadExplainabilityCharts() {
        try {
            const resp = await fetch("/api/analytics/explainability");
            const json = await resp.json();
            if (!json.success) return;
            const data = json.data;

            // LightGBM Feature Importance Chart
            destroyChart("lgbImportance");
            const lCtx = document.getElementById("lgbImportanceChart");
            if (lCtx && data.lightgbm_feature_importance.length) {
                const top10 = data.lightgbm_feature_importance.slice(0, 8);
                chartInstances["lgbImportance"] = new Chart(lCtx, {
                    type: "bar",
                    data: {
                        labels: top10.map(i => i.feature),
                        datasets: [{
                            label: "Split / Gain Score",
                            data: top10.map(i => i.importance_score),
                            backgroundColor: "#10b981",
                            borderRadius: 6
                        }]
                    },
                    options: {
                        indexAxis: "y",
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } },
                            y: { grid: { display: false }, ticks: { color: "#f8fafc" } }
                        }
                    }
                });
            }

            // Linear Regression Coefficients Chart
            destroyChart("lrCoef");
            const lrCtx = document.getElementById("lrCoefChart");
            if (lrCtx && data.linear_regression_top_coefficients.length) {
                const topCoefs = data.linear_regression_top_coefficients.slice(0, 8);
                chartInstances["lrCoef"] = new Chart(lrCtx, {
                    type: "bar",
                    data: {
                        labels: topCoefs.map(c => c.feature),
                        datasets: [{
                            label: "Standardized Beta Weight",
                            data: topCoefs.map(c => c.coefficient),
                            backgroundColor: topCoefs.map(c => c.coefficient >= 0 ? "#06b6d4" : "#f43f5e"),
                            borderRadius: 6
                        }]
                    },
                    options: {
                        indexAxis: "y",
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } },
                            y: { grid: { display: false }, ticks: { color: "#f8fafc" } }
                        }
                    }
                });
            }

        } catch (e) {
            console.error("Explainability error:", e);
        }
    }

    // Utility: Debounce
    function debounce(func, wait) {
        let timeout;
        return function (...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // Initialize application
    initApp();
});
