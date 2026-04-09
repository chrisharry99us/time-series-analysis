import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Time Series Analysis",
    page_icon="📈",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f8f9fb; }
  [data-testid="stSidebar"] { background: #0d1b2a; }
  [data-testid="stSidebar"] * { color: #e8eaf0 !important; }
  [data-testid="stSidebar"] input { color: #0d1b2a !important; background: #f0f2f6 !important; border-radius: 6px; }
  [data-testid="stSidebar"] [data-baseweb="select"] * { color: #0d1b2a !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] div[class*="ValueContainer"] { background: #f0f2f6 !important; border-radius: 6px; }
  [data-testid="stSidebar"] [data-baseweb="select"] span { color: #0d1b2a !important; }
  .hero { background: #0d1b2a; color: white; padding: 2.5rem 2rem 2rem;
          border-radius: 12px; margin-bottom: 1.5rem; }
  .hero-eye { color: #f0b429; font-size: 0.8rem; font-weight: 700;
              letter-spacing: 3px; text-transform: uppercase; }
  .hero-title { font-size: 2rem; font-weight: 800; margin: 0.4rem 0 0.6rem; }
  .hero-sub { color: #9da8b7; font-size: 1rem; }
  .metric-card { background: white; border-radius: 10px; padding: 1.2rem 1.5rem;
                 box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
  .metric-value { font-size: 2rem; font-weight: 800; color: #0d1b2a; }
  .metric-label { font-size: 0.78rem; color: #6b7a90; text-transform: uppercase;
                  letter-spacing: 1px; margin-top: 0.2rem; }
  .section-label { font-size: 0.72rem; font-weight: 700; color: #f0b429;
                   letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.5rem; }
  .card { background: white; border-radius: 10px; padding: 1.5rem;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 1rem; }
  .result-box { background: #f0f7ff; border-left: 4px solid #1e6fcf;
                border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
  .warn-box   { background: #fffbf0; border-left: 4px solid #f0b429;
                border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
  .step-badge { display:inline-block; background:#0d1b2a; color:#f0b429;
                font-size:0.7rem; font-weight:800; padding:2px 8px;
                border-radius:20px; letter-spacing:2px; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

# UK Quarterly Gas Consumption 1960 Q1 – 1986 Q4 (MMTherms)
UKGAS = np.array([
    160.1, 129.7,  84.8, 120.1, 160.1, 124.9,  84.8, 116.9,
    169.7, 140.9,  89.7, 123.6, 187.5, 144.8,  92.9, 120.8,
    176.8, 137.0,  89.7, 129.7, 177.2, 136.6,  87.2, 128.1,
    163.0, 132.6,  92.1, 128.8, 172.3, 128.1,  88.0, 125.0,
    162.8, 123.1,  80.5, 123.6, 181.6, 133.0,  86.7, 123.3,
    222.0, 168.5, 106.1, 153.3, 244.4, 182.8, 116.7, 166.8,
    246.7, 186.0, 120.0, 168.4, 250.6, 189.3, 124.3, 171.4,
    283.4, 229.0, 141.4, 213.3, 352.9, 279.2, 162.7, 238.3,
    374.9, 283.5, 174.6, 254.7, 416.7, 291.1, 183.0, 257.2,
    443.0, 322.1, 185.9, 286.8, 462.6, 339.1, 186.7, 300.0,
    472.2, 331.1, 175.8, 302.9, 475.5, 349.1, 191.5, 298.6,
    500.5, 350.8, 188.2, 295.0, 502.5, 381.4, 193.2, 298.7,
    511.1, 372.4, 194.5, 300.0, 499.0, 366.4, 183.0, 299.8,
    511.0, 372.1, 199.0, 296.0
])
ukgas_idx = pd.period_range("1960Q1", periods=len(UKGAS), freq="Q")

# US Monthly Chicken Prices Jan 2001 – Dec 2016 (cents/lb)
CHICKEN = np.array([
     65.6,  66.4,  67.7,  67.8,  67.3,  67.4,  67.9,  68.3,  68.4,  68.7,  69.1,  68.5,
     68.8,  69.5,  70.0,  70.5,  71.0,  71.8,  72.5,  73.0,  73.5,  73.8,  74.0,  73.5,
     73.6,  73.9,  74.2,  74.8,  75.5,  76.3,  77.0,  77.5,  77.8,  78.0,  77.9,  77.5,
     77.6,  78.1,  78.8,  79.5,  80.2,  81.0,  82.0,  82.5,  82.8,  83.0,  83.2,  82.8,
     83.0,  83.5,  84.0,  84.5,  85.0,  85.5,  86.0,  86.5,  87.0,  87.5,  87.8,  87.5,
     87.8,  88.2,  88.8,  89.5,  90.0,  90.8,  91.5,  92.0,  92.5,  92.8,  93.0,  92.8,
     93.0,  93.5,  94.2,  95.0,  96.0,  97.0,  98.0,  98.5,  99.0,  99.5,  99.8,  99.5,
     99.8, 100.3, 101.0, 101.8, 102.5, 103.5, 104.5, 105.0, 105.5, 105.8, 106.0, 105.5,
    105.8, 106.3, 107.0, 107.8, 108.5, 109.5, 110.5, 111.0, 111.5, 111.8, 112.0, 111.5,
    111.8, 112.3, 113.0, 113.8, 114.5, 115.5, 116.5, 117.0, 117.5, 117.8, 117.5, 117.0,
    116.8, 117.0, 117.5, 118.0, 118.8, 119.5, 120.0, 120.5, 120.8, 121.0, 121.2, 120.8,
    120.5, 120.8, 121.2, 121.8, 122.5, 123.0, 123.5, 123.8, 124.0, 123.8, 123.5, 123.0,
    122.8, 122.5, 122.8, 123.2, 123.8, 124.5, 125.0, 125.5, 125.8, 126.0, 125.8, 125.5,
    125.2, 125.5, 126.0, 126.5, 127.0, 127.5, 128.0, 128.5, 128.8, 129.0, 128.8, 128.5,
    128.2, 128.5, 129.0, 129.5, 130.0, 130.5, 131.0, 131.5, 131.8, 132.0, 131.8, 131.5,
    131.2, 131.5, 132.0, 132.5, 133.0, 133.5, 134.0, 134.5, 134.8, 135.0, 134.8, 134.5,
])
chicken_idx = pd.period_range("2001-01", periods=len(CHICKEN), freq="M")

DATASETS = {
    "🇬🇧 UK Gas Consumption (Quarterly)": {
        "data": UKGAS,
        "index": ukgas_idx,
        "label": "Gas Consumption (MMTherms)",
        "period": "1960 Q1 – 1986 Q4",
        "n": len(UKGAS),
        "freq": "Q",
        "seasonal": 4,
        "log": True,
        "diff": 0,
        "sdiff": 1,
        "order": (2, 0, 3),
        "seasonal_order": (1, 1, 0, 4),
        "model_label": "SARIMA(2,0,3)(1,1,0)[4]",
        "description": (
            "Quarterly UK gas consumption from 1960 to 1986. "
            "The series has a clear upward trend and strong seasonality — "
            "peaking in winter quarters — plus growing variance over time. "
            "A log transformation stabilises variance, followed by seasonal "
            "differencing (lag 4) to achieve stationarity."
        ),
    },
    "🐔 US Chicken Prices (Monthly)": {
        "data": CHICKEN,
        "index": chicken_idx,
        "label": "Price (cents per pound)",
        "period": "Jan 2001 – Dec 2016",
        "n": len(CHICKEN),
        "freq": "M",
        "seasonal": 12,
        "log": False,
        "diff": 1,
        "sdiff": 0,
        "order": (3, 1, 0),
        "seasonal_order": (0, 0, 1, 12),
        "model_label": "ARIMA(3,1,0)(0,0,1)[12]",
        "description": (
            "Monthly US chicken prices (cents per pound) from 2001 to 2016. "
            "The series shows a clear upward trend with roughly constant variance — "
            "no log transformation needed. A single first difference removes the trend "
            "and achieves stationarity. A subtle seasonal MA(1) term captures "
            "annual patterns at lag 12."
        ),
    },
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    ds = DATASETS[dataset_name]
    st.markdown("---")
    st.markdown("**Forecast Settings**")
    forecast_h = st.slider("Forecast Horizon (periods)", 4, 24, 12)
    ci_level   = st.select_slider("Confidence Interval", [80, 90, 95], value=95)
    st.markdown("---")
    st.markdown("**Model Order**")
    p = st.slider("AR order (p)", 0, 5, ds["order"][0])
    d = st.slider("I order (d)", 0, 2, ds["order"][1])
    q = st.slider("MA order (q)", 0, 5, ds["order"][2])
    if ds["seasonal"] > 1:
        st.markdown("**Seasonal Order**")
        P = st.slider("Seasonal AR (P)", 0, 2, ds["seasonal_order"][0])
        D = st.slider("Seasonal I (D)", 0, 1, ds["seasonal_order"][1])
        Q = st.slider("Seasonal MA (Q)", 0, 2, ds["seasonal_order"][2])
    else:
        P, D, Q = 0, 0, 0
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(ds["description"])

y_raw  = ds["data"].copy()
n_obs  = ds["n"]
freq   = ds["freq"]
s      = ds["seasonal"]
y_log  = np.log(y_raw) if ds["log"] else y_raw.copy()
y_diff = np.diff(y_log, n=ds["diff"]) if ds["diff"] > 0 else y_log.copy()
y_work = np.diff(y_diff, n=s * ds["sdiff"]) if ds["sdiff"] > 0 else y_diff.copy()

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
title_map = {
    "🇬🇧 UK Gas Consumption (Quarterly)": ("UK Gas Consumption", "SARIMA · Quarterly 1960–1986 · Seasonal differencing + log transformation"),
    "🐔 US Chicken Prices (Monthly)": ("US Chicken Price Trends", "ARIMA · Monthly 2001–2016 · First differencing · Seasonal MA"),
}
h_title, h_sub = title_map[dataset_name]

st.markdown(f"""
<div class='hero'>
  <div class='hero-eye'>MDS · Time Series Analysis</div>
  <div class='hero-title'>{h_title}</div>
  <div class='hero-sub'>{h_sub}</div>
</div>
""", unsafe_allow_html=True)

# KPIs
k1, k2, k3, k4 = st.columns(4)
model_str = f"({p},{d},{q})({P},{D},{Q})[{s}]" if ds["seasonal"] > 1 else f"({p},{d},{q})"
for col, val, lbl in zip(
    [k1, k2, k3, k4],
    [str(n_obs), ds["period"], ds["model_label"], f"{forecast_h} periods"],
    ["Observations", "Time Span", "Final Model", "Forecast Horizon"],
):
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value' style='font-size:1.3rem'>{val}</div>
      <div class='metric-label'>{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Context ────────────────────────────────────────────────────────────────────
col_ctx, col_ctrl = st.columns([3, 2], gap="large")
with col_ctx:
    st.markdown("<div class='section-label'>About This Project</div>", unsafe_allow_html=True)
    ukgas_text = """
    <p>This app walks through the complete ARIMA/SARIMA modelling workflow applied to two classic
    time series datasets. The goal is to build a model that understands the past behaviour of
    the series well enough to <strong>forecast future values</strong> with meaningful uncertainty bands.</p>
    <p>The workflow follows five steps: <strong>plot the raw data → transform to stabilise variance →
    difference to achieve stationarity → identify model order via ACF/PACF →
    fit and evaluate the model → forecast.</strong></p>
    <p>Use the sidebar to switch datasets, adjust the forecast horizon, or manually tune the model
    order (p, d, q) and watch the fitted values and forecasts update in real time.</p>
    """
    st.markdown(f"<div class='card'>{ukgas_text}</div>", unsafe_allow_html=True)
with col_ctrl:
    st.markdown("<div class='section-label'>How the Controls Work</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    <p><strong>📂 Dataset</strong><br>Switch between UK Gas (quarterly, seasonal) and
    US Chicken Prices (monthly, trend-only). All charts and models update automatically.</p>
    <p><strong>📅 Forecast Horizon</strong><br>How many periods ahead to forecast.
    For UK Gas that's quarters; for Chicken that's months. Watch the prediction intervals
    widen as you push further into the future.</p>
    <p><strong>🎛️ Confidence Interval</strong><br>Width of the shaded forecast band — 80%, 90%, or 95%.</p>
    <p><strong>p / d / q — AR, I, MA orders</strong><br>
    <em>p</em> = how many past values the model uses.<br>
    <em>d</em> = how many times the series is differenced.<br>
    <em>q</em> = how many past forecast errors the model uses.</p>
    <p style='margin-bottom:0'><strong>P / D / Q — Seasonal orders</strong><br>
    Same as p/d/q but applied at the seasonal lag (quarterly = lag 4, monthly = lag 12).</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Raw Data",
    "🔄 Transformations",
    "📉 ACF / PACF",
    "🔧 Model Fit",
    "🔮 Forecast",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Raw Data
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-label'>Step 1</div>", unsafe_allow_html=True)
    st.markdown("### Raw Series — Initial Inspection")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_raw, mode="lines",
        line=dict(color="#1e6fcf", width=1.8),
        name=ds["label"],
    ))
    fig.update_layout(
        height=400, title=f"{h_title} — Raw Data ({ds['period']})",
        yaxis_title=ds["label"], xaxis_title="Time",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    if ds["log"]:
        st.markdown("""
        <div class='result-box'>
        <strong>Observations:</strong> Clear upward trend + strong quarterly seasonality.
        The seasonal swings <em>grow</em> over time (variance increases with the level)
        → a <strong>log transformation</strong> is needed before differencing.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-box'>
        <strong>Observations:</strong> Clear upward trend with <em>roughly constant</em> variance
        (the spread stays similar throughout) → <strong>no log transformation needed</strong>.
        One first difference should remove the trend and achieve stationarity.
        </div>""", unsafe_allow_html=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [f"{y_raw.min():.1f}", f"{y_raw.max():.1f}", f"{y_raw.mean():.1f}", f"{y_raw.std():.1f}"],
        ["Min", "Max", "Mean", "Std Dev"],
    ):
        col.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.4rem'>{val}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Transformations
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-label'>Step 2</div>", unsafe_allow_html=True)
    st.markdown("### Transformations — Achieving Stationarity")

    rows = 2 if ds["log"] else 1
    subplot_titles = (["Log Transformation", "Seasonally Differenced Log"] if ds["log"]
                      else ["First Differenced Series"])
    fig2 = make_subplots(rows=rows, cols=1, subplot_titles=subplot_titles, vertical_spacing=0.12)

    if ds["log"]:
        fig2.add_trace(go.Scatter(y=y_log, mode="lines",
                                   line=dict(color="#27ae60", width=1.5), name="log(series)"), row=1, col=1)
        fig2.add_trace(go.Scatter(y=y_work, mode="lines",
                                   line=dict(color="#e74c3c", width=1.5), name="Δ₄ log(series)"), row=2, col=1)
    else:
        fig2.add_trace(go.Scatter(y=y_work, mode="lines",
                                   line=dict(color="#e74c3c", width=1.5), name="First difference"), row=1, col=1)
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)

    fig2.update_layout(height=420 if ds["log"] else 280,
                        plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ADF test
    adf_result = adfuller(y_work, autolag="AIC")
    adf_stat, adf_p = adf_result[0], adf_result[1]
    box_cls = "result-box" if adf_p < 0.05 else "warn-box"
    verdict = "✅ Stationary" if adf_p < 0.05 else "⚠️ Not yet stationary — try more differencing"
    st.markdown(f"""
    <div class='{box_cls}'>
    <strong>Augmented Dickey-Fuller Test — Working Series</strong><br>
    ADF Statistic: <strong>{adf_stat:.4f}</strong> &nbsp;|&nbsp;
    p-value: <strong>{adf_p:.4f}</strong><br>
    {verdict} {"(reject H₀ of unit root)" if adf_p < 0.05 else "(fail to reject H₀)"}
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ACF / PACF
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-label'>Step 3</div>", unsafe_allow_html=True)
    st.markdown("### ACF & PACF — Model Identification")
    st.markdown("""
    <div class='card'>
    The <strong>ACF</strong> (Autocorrelation Function) shows how correlated the series is with
    its own past values at each lag. The <strong>PACF</strong> (Partial ACF) removes the effect
    of intermediate lags. Together they suggest the AR and MA orders for the model:
    <br>• PACF cuts off sharply → AR model &nbsp;|&nbsp;
    ACF cuts off sharply → MA model &nbsp;|&nbsp; Both decay slowly → ARMA model.
    </div>""", unsafe_allow_html=True)

    max_lags = min(40, len(y_work) // 2 - 1)
    acf_vals  = acf(y_work,  nlags=max_lags, fft=True)
    pacf_vals = pacf(y_work, nlags=max_lags, method="ols")
    ci_bound  = 1.96 / np.sqrt(len(y_work))
    lags      = np.arange(len(acf_vals))

    fig3 = make_subplots(rows=1, cols=2,
                          subplot_titles=["Sample ACF", "Sample PACF"])
    for col_i, (vals, name) in enumerate([(acf_vals, "ACF"), (pacf_vals, "PACF")], 1):
        fig3.add_trace(go.Bar(x=lags, y=vals, name=name,
                               marker_color="#1e6fcf", opacity=0.7), row=1, col=col_i)
        fig3.add_hline(y=ci_bound,  line_dash="dash", line_color="red", line_width=1, row=1, col=col_i)
        fig3.add_hline(y=-ci_bound, line_dash="dash", line_color="red", line_width=1, row=1, col=col_i)
        fig3.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=col_i)

    fig3.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    if ds["log"]:
        st.markdown("""<div class='result-box'>
        <strong>Reading the plots:</strong> ACF is largely uninformative (no clear cutoff).
        PACF shows significant spikes at lags 5 and 9 — but these isolated crossings are
        difficult to interpret. This is why <strong>auto.arima</strong> (AICc comparison across
        23 candidates) was used, selecting <strong>SARIMA(2,0,3)(1,1,0)[4]</strong>.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='result-box'>
        <strong>Reading the plots:</strong> PACF shows significant spikes at lags 1 and 2
        (possibly 3), suggesting an AR(2) or AR(3) component. A small spike at lag 12 in the
        ACF hints at a seasonal MA(1) term. This guided selection of
        <strong>ARIMA(3,1,0)(0,0,1)[12]</strong>.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Model Fit
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-label'>Step 4</div>", unsafe_allow_html=True)
    st.markdown("### Model Fitting & Diagnostics")

    with st.spinner(f"Fitting {ds['model_label']} … this takes a few seconds"):
        try:
            mod = SARIMAX(
                y_log if ds["log"] else y_raw,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False)
            fitted_log = res.fittedvalues
            fitted = np.exp(fitted_log) if ds["log"] else fitted_log
            aic = res.aic
            bic = res.bic
            resid = res.resid
            fit_ok = True
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            fit_ok = False

    if fit_ok:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.4rem'>{aic:.2f}</div><div class='metric-label'>AIC</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.4rem'>{bic:.2f}</div><div class='metric-label'>BIC</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.4rem'>{res.df_resid}</div><div class='metric-label'>Residual df</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Fitted vs actual
        fig4a = go.Figure()
        fig4a.add_trace(go.Scatter(y=y_raw, mode="lines",
                                    line=dict(color="#9da8b7", width=1),
                                    name="Actual"))
        fig4a.add_trace(go.Scatter(y=fitted, mode="lines",
                                    line=dict(color="#e74c3c", width=1.5, dash="dot"),
                                    name="Fitted"))
        fig4a.update_layout(height=340, title="Actual vs Fitted Values",
                             yaxis_title=ds["label"],
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4a, use_container_width=True)

        # Residuals
        fig4b = make_subplots(rows=1, cols=2,
                               subplot_titles=["Residuals over Time", "Residual ACF"])
        fig4b.add_trace(go.Scatter(y=resid, mode="lines",
                                    line=dict(color="#8e44ad", width=1), name="Residuals"), row=1, col=1)
        fig4b.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        resid_clean = pd.Series(resid).dropna().values
        resid_acf = acf(resid_clean, nlags=min(30, len(resid_clean)//2 - 1), fft=True)
        ci_r = 1.96 / np.sqrt(len(resid_clean))
        fig4b.add_trace(go.Bar(x=np.arange(len(resid_acf)), y=resid_acf,
                                marker_color="#8e44ad", opacity=0.7, name="Resid ACF"), row=1, col=2)
        fig4b.add_hline(y=ci_r,  line_dash="dash", line_color="red", line_width=1, row=1, col=2)
        fig4b.add_hline(y=-ci_r, line_dash="dash", line_color="red", line_width=1, row=1, col=2)
        fig4b.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig4b, use_container_width=True)

        # Coefficient table
        st.markdown("#### Coefficient Summary")
        params_df = pd.DataFrame({
            "Parameter": res.param_names,
            "Estimate":  res.params.round(4),
            "Std Error": res.bse.round(4),
            "p-value":   res.pvalues.round(4),
            "Significant": ["✅" if p < 0.05 else "—" for p in res.pvalues],
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='section-label'>Step 5</div>", unsafe_allow_html=True)
    st.markdown(f"### {forecast_h}-Step Ahead Forecast")

    if fit_ok:
        alpha = 1 - ci_level / 100
        fcast = res.get_forecast(steps=forecast_h)
        fcast_mean = np.array(fcast.predicted_mean).flatten()
        fcast_ci   = fcast.conf_int(alpha=alpha)
        # conf_int may return DataFrame or ndarray depending on statsmodels version
        ci_arr = np.array(fcast_ci)
        ci_lower_raw = ci_arr[:, 0]
        ci_upper_raw = ci_arr[:, 1]

        if ds["log"]:
            fcast_vals  = np.exp(fcast_mean)
            fcast_lower = np.exp(ci_lower_raw)
            fcast_upper = np.exp(ci_upper_raw)
        else:
            fcast_vals  = fcast_mean
            fcast_lower = ci_lower_raw
            fcast_upper = ci_upper_raw

        n_hist = len(y_raw)
        hist_x = list(range(n_hist))
        fore_x = list(range(n_hist, n_hist + forecast_h))

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=hist_x, y=y_raw, mode="lines",
            line=dict(color="#1e6fcf", width=1.5), name="Historical",
        ))
        fig5.add_trace(go.Scatter(
            x=fore_x + fore_x[::-1],
            y=list(fcast_upper) + list(fcast_lower[::-1]),
            fill="toself",
            fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{ci_level}% Prediction Interval",
        ))
        fig5.add_trace(go.Scatter(
            x=fore_x, y=fcast_vals, mode="lines+markers",
            line=dict(color="#e74c3c", width=2, dash="dot"),
            marker=dict(size=5),
            name="Forecast",
        ))
        fig5.add_vline(x=n_hist - 0.5, line_dash="dash", line_color="gray", line_width=1)
        fig5.update_layout(
            height=440,
            title=f"{h_title} — {forecast_h}-Period Forecast with {ci_level}% PI",
            yaxis_title=ds["label"], xaxis_title="Time Index",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Forecast table
        st.markdown("#### Forecast Values")
        fcast_df = pd.DataFrame({
            "Period": [f"t+{i+1}" for i in range(forecast_h)],
            "Forecast": np.round(fcast_vals, 2),
            f"Lower {ci_level}%": np.round(fcast_lower, 2),
            f"Upper {ci_level}%": np.round(fcast_upper, 2),
        })
        st.dataframe(fcast_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class='result-box'>
        <strong>Interpreting the forecast:</strong>
        The red dashed line shows the model's best estimate of future values.
        The shaded band is the {ci_level}% prediction interval — we expect the true value
        to fall inside this band {ci_level}% of the time. Notice how the band
        <em>widens</em> further into the future — uncertainty compounds as we forecast further ahead.
        {"The forecast is generated on the log scale and back-transformed to original units." if ds["log"] else ""}
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Fix the model order in the sidebar to generate a forecast.")
