// ForkCast AI – Dashboard JS
// All API calls go to the FastAPI backend on the same origin.

const API = "";  // same-origin

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
  predictions: [],
  recommendations: [],
  summary: {},
  ingredients: [],
  currentIngId: null,
  charts: {},
};

// ── Helpers ───────────────────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch(API + path);
  if (r.status === 202) return { __pending: true, detail: (await r.json()).detail };
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function post(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function put(path, body) {
  const r = await fetch(API + path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function fmt(n, dec = 0) {
  return typeof n === "number" ? n.toFixed(dec) : "–";
}

function fmtCurrency(n) {
  return "$" + (n || 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

// ── Training status poller ────────────────────────────────────────────────────
let _pollTimer = null;
function startStatusPoll() {
  _pollTimer = setInterval(async () => {
    const s = await api("/api/status").catch(() => null);
    if (!s) return;
    const badge = document.getElementById("train-status");
    if (s.trained && !s.training) {
      badge.textContent = "✅ Models Ready";
      badge.className = "status-badge ready";
      clearInterval(_pollTimer);
      App.loadDashboard();
    } else if (s.training) {
      badge.textContent = "⏳ Training…";
      badge.className = "status-badge training";
    }
  }, 2500);
}

// ── Navigation ────────────────────────────────────────────────────────────────
function setTab(tab) {
  document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelector(`[data-tab="${tab}"]`).classList.add("active");
  document.getElementById(`tab-${tab}`).classList.add("active");

  const titles = {
    dashboard:   ["Dashboard",   "Overview & tomorrow's forecast"],
    predictions: ["Predictions", "Next-day sales forecast per item"],
    inventory:   ["Inventory",   "Ingredient purchase recommendations"],
    history:     ["Performance", "Model accuracy & historical comparison"],
    settings:    ["Settings",    "Stock levels, weather, promotions"],
  };
  document.getElementById("page-title").textContent = titles[tab][0];
  document.getElementById("page-subtitle").textContent = titles[tab][1];

  if (tab === "history")   App.loadPerformance();
  if (tab === "settings")  App.loadSettings();
  if (tab === "inventory") App.loadInventory();
  if (tab === "predictions") App.loadPredictions();
}

// ── App object ────────────────────────────────────────────────────────────────
const App = {

  // ── Dashboard ──────────────────────────────────────────────────────────────
  async loadDashboard() {
    const [recData, trendData] = await Promise.all([
      api("/api/inventory/recommendations").catch(() => null),
      api("/api/sales/daily-totals?days=90").catch(() => []),
    ]);

    // Weather strip loads independently so a network timeout doesn't block the page
    this.loadWeather();

    if (!recData || recData.__pending) {
      document.getElementById("summary-cards").innerHTML =
        `<div style="color:#64748b;padding:20px">${recData?.detail || "Loading predictions…"}</div>`;
      return;
    }

    state.predictions    = recData.predictions;
    state.recommendations = recData.recommendations;
    state.summary        = recData.summary;

    this._renderSummaryCards(recData.summary);
    this._renderPredictionsChart(recData.predictions);
    this._renderCategoryChart(recData.predictions);
    this._renderTrendChart(trendData);
    this._renderAlerts(recData.recommendations);

    // Update date label
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    document.getElementById("pred-date-label").textContent =
      "Predicting for " + tomorrow.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" });
  },

  _renderSummaryCards(s) {
    const cards = [
      { label: "Predicted Orders", value: s.total_predicted_orders, sub: "tomorrow", color: "blue", icon: "🍽️" },
      { label: "Predicted Revenue", value: fmtCurrency(s.predicted_revenue), sub: "gross", color: "green", icon: "💰" },
      { label: "Order Cost", value: fmtCurrency(s.total_order_cost), sub: "ingredients to buy", color: "amber", icon: "🛒" },
      { label: "Stockout Risks", value: s.stockout_risks, sub: "critical ingredients", color: s.stockout_risks > 0 ? "red" : "green", icon: "⚠️" },
      { label: "Items to Order", value: s.items_to_order, sub: "ingredient lines", color: "blue", icon: "📦" },
      { label: "Waste Savings Est.", value: fmtCurrency(s.waste_savings_est), sub: "vs no planning", color: "green", icon: "♻️" },
    ];
    document.getElementById("summary-cards").innerHTML = cards.map(c => `
      <div class="metric-card ${c.color}">
        <div class="metric-icon">${c.icon}</div>
        <div class="metric-label">${c.label}</div>
        <div class="metric-value">${c.value}</div>
        <div class="metric-sub">${c.sub}</div>
      </div>`).join("");
  },

  _renderPredictionsChart(preds) {
    destroyChart("predictions");
    const sorted = [...preds].sort((a, b) => b.predicted_qty - a.predicted_qty);
    const labels = sorted.map(p => p.item_name.replace(" & ", " &\n"));
    const data   = sorted.map(p => p.predicted_qty);
    const lower  = sorted.map(p => p.lower_bound);
    const upper  = sorted.map(p => p.upper_bound);

    const ctx = document.getElementById("chart-predictions").getContext("2d");
    state.charts.predictions = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Predicted Qty",
            data,
            backgroundColor: "rgba(59,130,246,0.8)",
            borderRadius: 6,
          },
          {
            label: "CI Lower",
            data: lower,
            type: "line",
            borderColor: "rgba(239,68,68,0.6)",
            borderDash: [4, 3],
            pointRadius: 3,
            fill: false,
          },
          {
            label: "CI Upper",
            data: upper,
            type: "line",
            borderColor: "rgba(34,197,94,0.6)",
            borderDash: [4, 3],
            pointRadius: 3,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales: { y: { beginAtZero: true, title: { display: true, text: "Units" } } },
      },
    });
  },

  _renderCategoryChart(preds) {
    destroyChart("category");
    const catMap = {};
    preds.forEach(p => {
      catMap[p.category] = (catMap[p.category] || 0) + p.predicted_qty;
    });
    const colors = { main: "#3b82f6", appetizer: "#f59e0b", dessert: "#ec4899", beverage: "#22c55e" };
    const ctx = document.getElementById("chart-category").getContext("2d");
    state.charts.category = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: Object.keys(catMap).map(k => k.charAt(0).toUpperCase() + k.slice(1)),
        datasets: [{
          data: Object.values(catMap),
          backgroundColor: Object.keys(catMap).map(k => colors[k] || "#94a3b8"),
          borderWidth: 2,
          borderColor: "#fff",
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "bottom" } },
        cutout: "60%",
      },
    });
  },

  _renderTrendChart(data) {
    destroyChart("trend");
    if (!data || !data.length) return;
    const ctx = document.getElementById("chart-trend").getContext("2d");
    state.charts.trend = new Chart(ctx, {
      type: "line",
      data: {
        labels: data.map(d => d.date),
        datasets: [{
          label: "Daily Revenue ($)",
          data: data.map(d => d.revenue),
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,0.1)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxTicksLimit: 12 } },
          y: { ticks: { callback: v => "$" + v.toLocaleString() } },
        },
      },
    });
  },

  _renderAlerts(recs) {
    const critical = recs.filter(r => r.status === "critical");
    const low      = recs.filter(r => r.status === "low");
    let html = "";
    if (!critical.length && !low.length) {
      html = `<div class="alert-item alert-ok">✅ All ingredient levels are healthy for tomorrow's demand.</div>`;
    }
    critical.forEach(r => {
      html += `<div class="alert-item alert-critical">🚨 <strong>${r.name}</strong> – Current stock ${r.current_stock} ${r.unit} but need ${r.gross_required} ${r.unit}. Order immediately!</div>`;
    });
    low.forEach(r => {
      html += `<div class="alert-item alert-low">⚠️ <strong>${r.name}</strong> – Running low (${r.current_stock} ${r.unit} on hand, need ${r.gross_required} ${r.unit}).</div>`;
    });
    document.getElementById("alerts-list").innerHTML = html;
  },

  // ── Predictions tab ────────────────────────────────────────────────────────
  async loadPredictions() {
    const dateInput = document.getElementById("pred-date-input");
    const dateParam = dateInput.value ? `?target_date=${dateInput.value}` : "";
    const data = await api(`/api/predictions/next-day${dateParam}`).catch(() => null);

    if (!data || data.__pending) {
      document.getElementById("predictions-body").innerHTML =
        `<tr><td colspan="10" style="text-align:center;padding:20px;color:#64748b">${data?.detail || "Loading…"}</td></tr>`;
      return;
    }

    state.predictions = data;
    const sorted = [...data].sort((a, b) => b.predicted_qty - a.predicted_qty);

    document.getElementById("predictions-body").innerHTML = sorted.map(p => {
      const revenue = (p.predicted_qty * p.base_price * (1 - p.promo_discount / 100)).toFixed(2);
      const promoTag = p.has_promotion
        ? `<span class="badge badge-promo">${p.promo_discount}% off</span>`
        : `<span style="color:#94a3b8">—</span>`;
      return `<tr>
        <td><strong>${p.item_name}</strong></td>
        <td>${p.category}</td>
        <td class="num"><strong>${fmt(p.predicted_qty, 1)}</strong></td>
        <td class="num" style="color:#64748b">${fmt(p.lower_bound, 1)}</td>
        <td class="num" style="color:#64748b">${fmt(p.upper_bound, 1)}</td>
        <td class="num">${fmtCurrency(p.base_price)}</td>
        <td class="num" style="color:#22c55e">${fmtCurrency(parseFloat(revenue))}</td>
        <td>${promoTag}</td>
        <td class="num" style="color:#64748b">${fmt(p.model_mae, 1)}</td>
        <td>
          <input class="override-input" id="ov-${p.item_id}" type="number" value="${Math.round(p.predicted_qty)}" min="0"/>
          <button class="btn btn-sm btn-secondary" onclick="App.applyOverride(${p.item_id})">Set</button>
        </td>
      </tr>`;
    }).join("");

    this._renderCIChart(sorted);
  },

  async applyOverride(itemId) {
    const val = parseFloat(document.getElementById(`ov-${itemId}`).value);
    if (isNaN(val) || val < 0) return;
    await post("/api/predictions/override", { item_id: itemId, predicted_qty: val });
    App.loadPredictions();
  },

  _renderCIChart(preds) {
    destroyChart("ci");
    const ctx = document.getElementById("chart-ci").getContext("2d");
    const labels = preds.map(p => p.item_name);
    state.charts.ci = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Predicted",
            data: preds.map(p => p.predicted_qty),
            backgroundColor: "rgba(59,130,246,0.75)",
            borderRadius: 4,
          },
          {
            label: "Range (Low–High)",
            data: preds.map(p => [p.lower_bound, p.upper_bound]),
            backgroundColor: "rgba(239,68,68,0.2)",
            borderColor: "rgba(239,68,68,0.5)",
            borderWidth: 1,
            type: "bar",
            barPercentage: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" }, tooltip: {
          callbacks: {
            label: (ctx) => {
              if (Array.isArray(ctx.raw)) return `CI: ${ctx.raw[0].toFixed(1)} – ${ctx.raw[1].toFixed(1)}`;
              return `Predicted: ${ctx.raw.toFixed(1)}`;
            }
          }
        }},
        scales: { y: { beginAtZero: true } },
      },
    });
  },

  // ── Inventory tab ──────────────────────────────────────────────────────────
  async loadInventory() {
    const data = await api("/api/inventory/recommendations").catch(() => null);
    if (!data || data.__pending) {
      document.getElementById("inventory-body").innerHTML =
        `<tr><td colspan="12" style="text-align:center;padding:20px">${data?.detail || "Loading…"}</td></tr>`;
      return;
    }

    state.recommendations = data.recommendations;
    state.summary = data.summary;

    // Inventory metric cards
    const s = data.summary;
    document.getElementById("inventory-cards").innerHTML = [
      { label: "Total Order Cost", value: fmtCurrency(s.total_order_cost), color: "amber", icon: "💳" },
      { label: "Items to Order", value: s.items_to_order, color: "blue", icon: "📋" },
      { label: "Stockout Risks", value: s.stockout_risks, color: s.stockout_risks ? "red" : "green", icon: "⚠️" },
      { label: "Waste Savings Est.", value: fmtCurrency(s.waste_savings_est), color: "green", icon: "♻️" },
    ].map(c => `
      <div class="metric-card ${c.color}">
        <div class="metric-icon">${c.icon}</div>
        <div class="metric-label">${c.label}</div>
        <div class="metric-value">${c.value}</div>
      </div>`).join("");

    document.getElementById("inventory-body").innerHTML = data.recommendations.map(r => {
      const statusBadge = {
        critical: `<span class="badge badge-critical">Critical</span>`,
        low:      `<span class="badge badge-low">Low</span>`,
        ok:       `<span class="badge badge-ok">OK</span>`,
      }[r.status];
      const shelfIcon = r.shelf_life_days <= 3 ? "🔴" : r.shelf_life_days <= 7 ? "🟡" : "🟢";
      return `<tr>
        <td>${statusBadge}</td>
        <td><strong>${r.name}</strong></td>
        <td>${r.unit}</td>
        <td class="num">${fmt(r.current_stock, 2)}</td>
        <td class="num">${fmt(r.gross_required, 2)}</td>
        <td class="num" style="color:#64748b">${fmt(r.safety_stock, 2)}</td>
        <td class="num"><strong>${fmt(r.total_required, 2)}</strong></td>
        <td class="num">${fmt(r.to_order, 2)}</td>
        <td class="num"><strong>${fmt(r.order_units, 2)}</strong></td>
        <td class="num" style="color:#059669">${fmtCurrency(r.estimated_cost)}</td>
        <td>${shelfIcon} ${r.shelf_life_days}d</td>
        <td>
          <button class="btn btn-sm btn-secondary" onclick="App.openStockModal(${r.ingredient_id}, '${r.name}', ${r.current_stock})">Update Stock</button>
        </td>
      </tr>`;
    }).join("");
  },

  exportInventoryCSV() {
    if (!state.recommendations.length) return;
    const headers = ["Ingredient","Unit","Current Stock","Gross Required","Safety Buffer","Total Required","To Order","Order Units","Est. Cost ($)","Shelf Life (days)","Status"];
    const rows = state.recommendations.map(r =>
      [r.name, r.unit, r.current_stock, r.gross_required, r.safety_stock,
       r.total_required, r.to_order, r.order_units, r.estimated_cost, r.shelf_life_days, r.status]
    );
    const csv = [headers, ...rows].map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `inventory_order_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
  },

  // ── Performance tab ────────────────────────────────────────────────────────
  async loadPerformance() {
    const days = document.getElementById("perf-days").value || 30;
    const data = await api(`/api/model/performance?last_n_days=${days}`).catch(() => null);
    if (!data || data.__pending) return;

    // Metrics table
    const gradeColor = (mae) => mae < 3 ? "🟢" : mae < 7 ? "🟡" : "🔴";
    document.getElementById("perf-body").innerHTML = data.metrics.map(m => `<tr>
      <td><strong>${m.item_name}</strong></td>
      <td class="num">${fmt(m.mae, 2)}</td>
      <td class="num">${fmt(m.rmse, 2)}</td>
      <td class="num">${m.n_days}</td>
      <td>${gradeColor(m.mae)} ${m.mae < 3 ? "Excellent" : m.mae < 7 ? "Good" : "Needs Improvement"}</td>
    </tr>`).join("");

    // Actual vs predicted chart – show top 4 items by volume
    destroyChart("avp");
    const top4Items = [...new Set(data.detail_rows.map(r => r.item_name))].slice(0, 4);
    const colors = ["#3b82f6","#22c55e","#f59e0b","#ec4899"];

    const grouped = {};
    data.detail_rows.forEach(r => {
      if (!grouped[r.item_name]) grouped[r.item_name] = { dates: [], actual: [], predicted: [] };
      grouped[r.item_name].dates.push(r.date);
      grouped[r.item_name].actual.push(r.actual);
      grouped[r.item_name].predicted.push(r.predicted);
    });

    const allDates = [...new Set(data.detail_rows.map(r => r.date))].sort();
    const datasets = [];

    top4Items.forEach((name, i) => {
      const g = grouped[name];
      if (!g) return;
      datasets.push({
        label: `${name} (actual)`,
        data: allDates.map(d => {
          const idx = g.dates.indexOf(d);
          return idx >= 0 ? g.actual[idx] : null;
        }),
        borderColor: colors[i],
        borderDash: [],
        pointRadius: 2,
        fill: false,
        tension: 0.3,
      });
      datasets.push({
        label: `${name} (pred)`,
        data: allDates.map(d => {
          const idx = g.dates.indexOf(d);
          return idx >= 0 ? g.predicted[idx] : null;
        }),
        borderColor: colors[i],
        borderDash: [5, 3],
        pointRadius: 0,
        fill: false,
        tension: 0.3,
        borderWidth: 1.5,
      });
    });

    const ctx = document.getElementById("chart-avp").getContext("2d");
    state.charts.avp = new Chart(ctx, {
      type: "line",
      data: { labels: allDates, datasets },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales: {
          x: { ticks: { maxTicksLimit: 15 } },
          y: { beginAtZero: true },
        },
      },
    });

    // Error distribution
    destroyChart("errors");
    const errors = data.detail_rows.map(r => r.error_pct);
    const buckets = [0,10,20,30,40,50,100];
    const counts = [];
    for (let i = 0; i < buckets.length - 1; i++) {
      counts.push(errors.filter(e => e >= buckets[i] && e < buckets[i+1]).length);
    }
    const ctx2 = document.getElementById("chart-errors").getContext("2d");
    state.charts.errors = new Chart(ctx2, {
      type: "bar",
      data: {
        labels: ["0–10%","10–20%","20–30%","30–40%","40–50%","50%+"],
        datasets: [{
          label: "# of predictions",
          data: counts,
          backgroundColor: ["#22c55e","#86efac","#fde68a","#fca5a5","#f87171","#ef4444"],
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } },
      },
    });
  },

  // ── Settings tab ───────────────────────────────────────────────────────────
  async loadSettings() {
    const [ings, promos, holidays, cfg] = await Promise.all([
      api("/api/ingredients"),
      api("/api/promotions"),
      api("/api/holidays"),
      api("/api/config").catch(() => ({})),
    ]);

    // Populate location form
    if (cfg.restaurant_name) document.getElementById("cfg-name").value = cfg.restaurant_name;
    if (cfg.latitude)        document.getElementById("cfg-lat").value  = cfg.latitude;
    if (cfg.longitude)       document.getElementById("cfg-lon").value  = cfg.longitude;
    if (cfg.timezone)        document.getElementById("cfg-tz").value   = cfg.timezone;

    // Set manual weather date default to tomorrow
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const wDate = document.getElementById("w-date");
    if (wDate) wDate.value = tomorrow.toISOString().slice(0,10);

    state.ingredients = ings;

    // Stock table
    document.getElementById("stock-body").innerHTML = ings.map(i => `<tr>
      <td><strong>${i.name}</strong></td>
      <td>${i.unit}</td>
      <td class="num">${fmt(i.current_stock, 2)}</td>
      <td><input type="number" id="st-${i.id}" value="${fmt(i.current_stock, 2)}" step="0.1" min="0" class="override-input"/></td>
      <td><button class="btn btn-sm btn-primary" onclick="App.saveStock(${i.id})">Save</button></td>
    </tr>`).join("");

    // Promotions
    document.getElementById("promos-list").innerHTML = promos.map(p => {
      const itemLabel = p.menu_item_id ? `Item #${p.menu_item_id}` : "All items";
      return `<div class="list-item">
        <div>
          <div class="list-item-name">${p.name}</div>
          <div class="list-item-detail">${itemLabel} · ${p.start_date} → ${p.end_date}</div>
        </div>
        <span class="badge badge-promo">${p.discount_pct}% off</span>
      </div>`;
    }).join("") || "<p style='color:#94a3b8;padding:10px 0'>No promotions configured.</p>";

    // Holidays
    document.getElementById("holidays-list").innerHTML = holidays.map(h => {
      const badge = h.impact_factor === 0
        ? `<span class="badge badge-closed">Closed</span>`
        : `<span class="badge badge-ok">${h.impact_factor}× traffic</span>`;
      return `<div class="list-item">
        <div>
          <div class="list-item-name">${h.name}</div>
          <div class="list-item-detail">${h.date}</div>
        </div>
        ${badge}
      </div>`;
    }).join("");
  },

  async saveStock(ingId) {
    const val = parseFloat(document.getElementById(`st-${ingId}`).value);
    if (isNaN(val) || val < 0) return;
    await put(`/api/ingredients/${ingId}/stock`, { quantity: val });
    // Subtle feedback
    const btn = event.target;
    btn.textContent = "✓ Saved";
    setTimeout(() => btn.textContent = "Save", 1500);
  },

  async saveWeather() {
    const dateVal = document.getElementById("w-date").value;
    const temp    = parseFloat(document.getElementById("w-temp").value);
    const precip  = parseFloat(document.getElementById("w-precip").value);
    const cond    = document.getElementById("w-condition").value;
    const msg     = document.getElementById("weather-msg");
    const params  = dateVal ? `?target_date=${dateVal}` : "";
    try {
      await post(`/api/weather/manual${params}`, { temperature: temp, precipitation: precip, condition: cond });
      msg.textContent = "✅ Override saved. Reload predictions to see effect.";
      msg.className = "msg ok";
      this.loadWeather();
    } catch {
      msg.textContent = "❌ Failed to save override.";
      msg.className = "msg err";
    }
  },

  // ── Weather strip ───────────────────────────────────────────────────────────
  async loadWeather() {
    const [forecast, cfg] = await Promise.all([
      api("/api/weather/forecast?days=7").catch(() => []),
      api("/api/config").catch(() => ({})),
    ]);

    // Location label
    const locEl = document.getElementById("weather-location");
    if (locEl && cfg.restaurant_name) {
      locEl.textContent = `— ${cfg.restaurant_name} (${cfg.latitude?.toFixed(4)}, ${cfg.longitude?.toFixed(4)})`;
    }

    const sourceEl = document.getElementById("weather-source");
    if (sourceEl) {
      sourceEl.textContent = forecast.length ? "Open-Meteo" : "No data";
      sourceEl.className = "source-tag " + (forecast.length ? "live" : "offline");
    }

    const strip = document.getElementById("weather-strip");
    if (!strip) return;

    if (!forecast.length) {
      strip.innerHTML = `<p style="color:#94a3b8;padding:8px">No forecast data. Click Refresh or check your internet connection.</p>`;
      return;
    }

    const days = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
    strip.innerHTML = forecast.map(f => {
      const d    = new Date(f.date + "T12:00:00");
      const day  = days[d.getDay()];
      const isToday = f.date === new Date().toISOString().slice(0,10);
      const isTomorrow = f.date === new Date(Date.now()+86400000).toISOString().slice(0,10);
      const label = isToday ? "Today" : isTomorrow ? "Tomorrow" : day;
      const precip = f.precipitation > 0
        ? `<span class="weather-precip">💧 ${f.precipitation}mm</span>` : "";
      return `<div class="weather-day ${isTomorrow ? "weather-tomorrow" : ""}">
        <div class="weather-label">${label}</div>
        <div class="weather-date">${f.date.slice(5)}</div>
        <div class="weather-emoji">${f.emoji}</div>
        <div class="weather-temp">${f.temperature}°C</div>
        <div class="weather-cond">${f.condition}</div>
        ${precip}
      </div>`;
    }).join("");
  },

  async refreshWeather() {
    const sourceEl = document.getElementById("weather-source");
    if (sourceEl) { sourceEl.textContent = "Fetching…"; sourceEl.className = "source-tag"; }
    try {
      const result = await post("/api/weather/refresh", {});
      this.loadWeather();
      const cfgMsg = document.getElementById("config-msg");
      if (cfgMsg) {
        cfgMsg.textContent = `✅ Fetched ${result.stored} days from Open-Meteo.`;
        cfgMsg.className = "msg ok";
      }
    } catch (e) {
      if (sourceEl) { sourceEl.textContent = "Offline"; sourceEl.className = "source-tag offline"; }
      const cfgMsg = document.getElementById("config-msg");
      if (cfgMsg) {
        cfgMsg.textContent = "❌ Could not reach Open-Meteo. Check your connection.";
        cfgMsg.className = "msg err";
      }
    }
  },

  async saveConfig() {
    const name = document.getElementById("cfg-name").value.trim();
    const lat  = parseFloat(document.getElementById("cfg-lat").value);
    const lon  = parseFloat(document.getElementById("cfg-lon").value);
    const tz   = document.getElementById("cfg-tz").value.trim();
    const msg  = document.getElementById("config-msg");

    if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      msg.textContent = "❌ Enter valid latitude (−90 to 90) and longitude (−180 to 180).";
      msg.className = "msg err";
      return;
    }

    msg.textContent = "⏳ Saving and fetching weather…";
    msg.className = "msg";

    try {
      const result = await post("/api/config", {
        restaurant_name: name || undefined,
        latitude:  lat,
        longitude: lon,
        timezone:  tz || undefined,
      });
      const stored = result.weather_updated ?? 0;
      const warn   = result.weather_warning ?? "";
      if (warn) {
        msg.textContent = `✅ Config saved. ⚠️ Weather: ${warn}`;
        msg.className = "msg ok";
      } else {
        msg.textContent = `✅ Config saved. Fetched ${stored} days of weather from Open-Meteo.`;
        msg.className = "msg ok";
      }
      this.loadWeather();
    } catch (e) {
      msg.textContent = "❌ Failed to save config.";
      msg.className = "msg err";
    }
  },

  // ── Stock modal ─────────────────────────────────────────────────────────────
  openStockModal(ingId, name, current) {
    state.currentIngId = ingId;
    document.getElementById("modal-ing-name").textContent = name;
    document.getElementById("modal-qty").value = current;
    document.getElementById("stock-modal").classList.remove("hidden");
  },

  closeModal() {
    document.getElementById("stock-modal").classList.add("hidden");
    state.currentIngId = null;
  },

  async confirmStockUpdate() {
    if (!state.currentIngId) return;
    const val = parseFloat(document.getElementById("modal-qty").value);
    await put(`/api/ingredients/${state.currentIngId}/stock`, { quantity: val });
    this.closeModal();
    this.loadInventory();
  },

  // ── Model retrain ───────────────────────────────────────────────────────────
  async retrain() {
    const badge = document.getElementById("train-status");
    badge.textContent = "⏳ Retraining…";
    badge.className = "status-badge training";
    await post("/api/model/retrain", {});
    startStatusPoll();
  },
};

// ── Boot ───────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Wire navigation
  document.querySelectorAll(".nav-item").forEach(el => {
    el.addEventListener("click", () => setTab(el.dataset.tab));
  });

  // Set default prediction date to tomorrow
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const iso = tomorrow.toISOString().slice(0, 10);
  document.getElementById("pred-date-input").value = iso;

  startStatusPoll();
});
