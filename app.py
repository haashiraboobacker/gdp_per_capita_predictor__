import streamlit as st
import joblib
import numpy as np
import json
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GDP Growth Predictor | Economic AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load model artefacts ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_artefacts():
    model  = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
    with open(os.path.join(BASE_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

try:
    model, scaler, meta = load_artefacts()
    MODEL_READY = True
except FileNotFoundError:
    MODEL_READY = False

FEATURE_COLS = [
    "year",
    "Inflation (CPI %)",
    "GDP (Current USD)",
    "GDP per Capita (Current USD)",
    "Unemployment Rate (%)",
    "Interest Rate (Real, %)",
    "Inflation (GDP Deflator, %)",
    "Current Account Balance (% GDP)",
    "Government Expense (% of GDP)",
    "Government Revenue (% of GDP)",
    "Tax Revenue (% of GDP)",
    "Gross National Income (USD)",
    "Public Debt (% of GDP)",
]

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #141428 40%, #0f1f3d 100%);
    min-height: 100vh;
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #1a1a40 0%, #2d1b69 50%, #0f3460 100%);
    border: 1px solid rgba(108, 99, 255, 0.35);
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(108,99,255,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero p { color: #94a3b8; font-size: 1.05rem; margin: 0; }
.hero .badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(108,99,255,0.2); border: 1px solid rgba(108,99,255,0.5);
    border-radius: 99px; padding: 4px 14px; font-size: 0.8rem; color: #a78bfa;
    margin-top: 1rem;
}

/* ── Section cards ── */
.card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(108,99,255,0.4); }
.card-title {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6c63ff; margin-bottom: 1rem;
}

/* ── Metric pills ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 1rem 0; }
.metric-pill {
    flex: 1; min-width: 140px;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px; padding: 0.75rem 1rem; text-align: center;
}
.metric-pill .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #a78bfa;
}
.metric-pill .lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

/* ── Prediction result box ── */
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 20px rgba(108,99,255,0.4); }
    50%      { box-shadow: 0 0 40px rgba(108,99,255,0.75), 0 0 80px rgba(52,211,153,0.2); }
}
.prediction-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(52,211,153,0.12));
    border: 2px solid rgba(108,99,255,0.6);
    border-radius: 20px; padding: 2.5rem; text-align: center;
    animation: pulse-glow 2.5s ease-in-out infinite;
    margin: 1.5rem 0;
}
.prediction-box .result-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.prediction-box .result-label {
    font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; letter-spacing: 0.05em;
}

/* ── Interpretation badge ── */
.interp {
    display: inline-block; border-radius: 99px;
    padding: 6px 20px; font-size: 0.9rem; font-weight: 600;
    margin-top: 1rem;
}
.interp-strong  { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
.interp-moderate{ background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }
.interp-weak    { background: rgba(239,68,68,0.2);   color: #ef4444; border: 1px solid #ef4444; }
.interp-negative{ background: rgba(239,68,68,0.25);  color: #f87171; border: 1px solid #f87171; }

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f2d 0%, #131330 100%) !important;
    border-right: 1px solid rgba(108,99,255,0.2);
}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] label { color: #cbd5e1 !important; font-size: 0.85rem !important; }

/* Slider accent */
[data-testid="stSlider"] [role="slider"] {
    background-color: #6c63ff !important;
}

/* ── Number input ── */
.stNumberInput input { background: rgba(255,255,255,0.06) !important; color: #e2e8f0 !important; border-radius: 8px !important; }

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6c63ff, #48c8a5) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important; font-size: 1rem !important;
    transition: transform 0.15s, opacity 0.15s !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-2px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Warning / info ── */
.not-ready {
    background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px; padding: 1.2rem; color: #f87171; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📈 Economic Growth Predictor</h1>
  <p>Powered by KNN Regression · World Bank Data 2025 · 180+ Countries</p>
  <div class="badge">🤖 &nbsp; KNN Regressor with StandardScaler &amp; SimpleImputer</div>
</div>
""", unsafe_allow_html=True)

# ─── Model status & metrics strip ───────────────────────────────────────────
if MODEL_READY:
    c1, c2, c3, c4 = st.columns(4)
    def metric_pill(col, val, label):
        col.markdown(f"""<div class="metric-pill"><div class="val">{val}</div><div class="lbl">{label}</div></div>""", unsafe_allow_html=True)

    metric_pill(c1, f"k={meta['best_k']}", "Optimal Neighbours")
    metric_pill(c2, f"{meta['r2_test']:.3f}", "Test R² Score")
    metric_pill(c3, f"{meta['mae_test']:.2f}%", "Mean Abs. Error")
    metric_pill(c4, f"{meta['rmse_test']:.2f}%", "RMSE")
else:
    st.markdown("""
    <div class="not-ready">
      ⚠️ <b>Model not found.</b> Please run <code>python train_model.py</code> first to generate
      <code>model.joblib</code> and <code>scaler.joblib</code>.
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Sidebar – Feature Inputs ────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
  <span style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
               font-weight:700; color:#a78bfa;">🌍 Input Economic Data</span><br>
  <span style="font-size:0.78rem; color:#64748b;">Adjust the sliders to match your country's indicators</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

def sidebar_section(title):
    st.sidebar.markdown(f"<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#6c63ff;margin-top:0.8rem;'>{title}</div>", unsafe_allow_html=True)

sidebar_section("📅 Time")
year = st.sidebar.slider("Year", min_value=2010, max_value=2030, value=2023, step=1)

sidebar_section("💵 Output & Income")
gdp = st.sidebar.number_input(
    "GDP (Current USD, Billions)", min_value=0.01, max_value=30000.0, value=500.0, step=0.1,
    help="Enter GDP in billions of USD. E.g. India ≈ 3500 B"
)
gdp_val = gdp * 1e9  # convert back to raw USD

gdp_per_capita = st.sidebar.number_input(
    "GDP per Capita (USD)", min_value=100.0, max_value=260000.0, value=5000.0, step=100.0
)
gni = st.sidebar.number_input(
    "Gross National Income (USD, Billions)", min_value=0.01, max_value=28000.0, value=490.0, step=0.1
)
gni_val = gni * 1e9

sidebar_section("📊 Inflation & Prices")
inflation_cpi = st.sidebar.slider(
    "Inflation – CPI (%)", min_value=-7.0, max_value=100.0, value=4.0, step=0.1,
    help="Consumer Price Index annual inflation rate"
)
inflation_gdp_def = st.sidebar.slider(
    "Inflation – GDP Deflator (%)", min_value=-30.0, max_value=100.0, value=4.5, step=0.1
)

sidebar_section("💼 Labour Market")
unemployment = st.sidebar.slider(
    "Unemployment Rate (%)", min_value=0.1, max_value=35.0, value=6.5, step=0.1
)

sidebar_section("💳 Finance & Trade")
interest_rate = st.sidebar.slider(
    "Real Interest Rate (%)", min_value=-82.0, max_value=62.0, value=4.0, step=0.1
)
current_account = st.sidebar.slider(
    "Current Account Balance (% of GDP)", min_value=-61.0, max_value=236.0, value=-2.0, step=0.1
)

sidebar_section("🏛️ Government Fiscal")
gov_expense = st.sidebar.slider(
    "Government Expense (% of GDP)", min_value=0.0, max_value=104.0, value=26.0, step=0.1
)
gov_revenue = st.sidebar.slider(
    "Government Revenue (% of GDP)", min_value=0.0, max_value=100.0, value=24.0, step=0.1
)
tax_revenue = st.sidebar.slider(
    "Tax Revenue (% of GDP)", min_value=0.0, max_value=50.0, value=16.0, step=0.1
)
public_debt = st.sidebar.slider(
    "Public Debt (% of GDP)", min_value=1.0, max_value=250.0, value=55.0, step=0.5
)

# ─── Predict button ──────────────────────────────────────────────────────────
predict_btn = st.sidebar.button("🔮  Predict GDP Growth", use_container_width=True)

# ─── Main layout ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card"><div class="card-title">📋 Your Economic Profile</div>', unsafe_allow_html=True)

    table_data = {
        "Indicator": [
            "Year", "GDP", "GDP per Capita", "GNI",
            "Inflation (CPI)", "Inflation (Deflator)", "Unemployment",
            "Interest Rate", "Current Account Balance",
            "Gov. Expense", "Gov. Revenue", "Tax Revenue", "Public Debt"
        ],
        "Value": [
            str(year),
            f"${gdp:,.1f} B",
            f"${gdp_per_capita:,.0f}",
            f"${gni:,.1f} B",
            f"{inflation_cpi:.1f}%",
            f"{inflation_gdp_def:.1f}%",
            f"{unemployment:.1f}%",
            f"{interest_rate:.1f}%",
            f"{current_account:.1f}% of GDP",
            f"{gov_expense:.1f}% of GDP",
            f"{gov_revenue:.1f}% of GDP",
            f"{tax_revenue:.1f}% of GDP",
            f"{public_debt:.1f}% of GDP",
        ]
    }

    import pandas as pd
    profile_df = pd.DataFrame(table_data)
    st.dataframe(
        profile_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Indicator": st.column_config.TextColumn("Indicator", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><div class="card-title">🎯 Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn:
        if not MODEL_READY:
            st.error("❌ Model files not found. Run `python train_model.py` first.")
        else:
            # Build input vector in the exact same order as training
            input_vector = np.array([[
                year,
                inflation_cpi,
                gdp_val,
                gdp_per_capita,
                unemployment,
                interest_rate,
                inflation_gdp_def,
                current_account,
                gov_expense,
                gov_revenue,
                tax_revenue,
                gni_val,
                public_debt,
            ]])

            # Scale with the fitted scaler (CRITICAL – same pipeline as training)
            input_scaled = scaler.transform(input_vector)

            # Predict
            prediction = model.predict(input_scaled)[0]

            # Choose interpretation badge
            if prediction >= 6.0:
                badge_class = "interp-strong"
                badge_text  = "🚀 Strong Growth"
                emoji = "🟢"
            elif prediction >= 2.5:
                badge_class = "interp-moderate"
                badge_text  = "📈 Moderate Growth"
                emoji = "🟡"
            elif prediction >= 0.0:
                badge_class = "interp-weak"
                badge_text  = "📊 Weak / Stagnant Growth"
                emoji = "🟠"
            else:
                badge_class = "interp-negative"
                badge_text  = "📉 Economic Contraction"
                emoji = "🔴"

            st.markdown(f"""
            <div class="prediction-box">
              <div style="font-size:0.85rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;">
                Predicted Annual GDP Growth
              </div>
              <div class="result-val">{prediction:+.2f}%</div>
              <div class="result-label">GDP Growth (% Annual) · KNN k={meta['best_k']}</div>
              <span class="interp {badge_class}">{emoji} {badge_text}</span>
            </div>
            """, unsafe_allow_html=True)

            # Contextual insights
            st.markdown("**💡 Key Insights for this Prediction:**")
            insights = []
            if inflation_cpi > 10:
                insights.append(f"⚠️ High inflation ({inflation_cpi:.1f}%) typically suppresses growth.")
            if unemployment > 12:
                insights.append(f"⚠️ High unemployment ({unemployment:.1f}%) reduces consumer demand.")
            if current_account < -10:
                insights.append(f"⚠️ Large current account deficit ({current_account:.1f}%) signals external vulnerability.")
            if public_debt > 100:
                insights.append(f"⚠️ Very high public debt ({public_debt:.0f}% GDP) may crowd out investment.")
            if prediction > 4 and inflation_cpi < 6:
                insights.append("✅ Low inflation + strong growth is a hallmark of a healthy economy.")
            if not insights:
                insights.append("📊 Indicators appear within typical global ranges.")
            for i in insights:
                st.markdown(f"- {i}")

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            text-align:center; padding: 4rem 2rem;
            border: 2px dashed rgba(108,99,255,0.3);
            border-radius: 16px; margin: 1rem 0;
        ">
          <div style="font-size: 3rem; margin-bottom: 1rem;">🔮</div>
          <div style="color: #6c63ff; font-size: 1.1rem; font-weight:600;">
            Adjust the sliders in the sidebar
          </div>
          <div style="color: #64748b; font-size: 0.9rem; margin-top:0.5rem;">
            then click <b style="color:#a78bfa;">Predict GDP Growth</b> to see the result
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── How it works section ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🧠 How This Model Works", expanded=False):
    st.markdown("""
    ### KNN Regression Pipeline

    | Step | What Happens | Why It Matters |
    |------|-------------|----------------|
    | **1. Target** | `GDP Growth (% Annual)` set as **y** | Ensures we predict the right variable |
    | **2. Features** | All 13 numerical columns (excluding target) | No leakage; no categorical noise |
    | **3. SimpleImputer** | Fill NaN with column **mean** | Handles ~2,620 missing Public Debt entries |
    | **4. StandardScaler** | Z-score normalise every feature | **Critical fix** — KNN uses Euclidean distance; without scaling, GDP (~10¹²) dominates everything |
    | **5. Best k** | 5-fold CV loop over k ∈ [1, 30] | Finds optimal neighbours; avoids over/underfitting |
    | **6. Export** | `model.joblib` + `scaler.joblib` | Reproducible, deployment-ready artefacts |

    **Root Cause of the Constant Prediction Bug:**
    > Without `StandardScaler`, the KNN algorithm computes distances purely based on
    > the raw feature magnitudes. GDP (Current USD) has values in the **trillions**,
    > while Inflation is in **single digits**. Every "nearest neighbour" was found by
    > GDP alone — making all predictions collapse to the mean. Scaling fixes this instantly.
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem; color: #334155; font-size: 0.8rem;">
  Built with ❤️ using Streamlit &amp; scikit-learn &nbsp;·&nbsp;
  Data: <a href="https://www.kaggle.com/datasets/ahmadfirman/world-economic-2025"
            style="color:#6c63ff; text-decoration:none;">World Bank 2025</a>
</div>
""", unsafe_allow_html=True)
