"""
CreditWise Loan System — app.py
LinkedIn-themed Streamlit app for Hugging Face Spaces
Author : Karthika Krishna M
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import json
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LendSure-AI · Loan Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LINKEDIN-INSPIRED GLOBAL CSS
# ─────────────────────────────────────────────
LINKEDIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ── */
:root {
    --li-blue-900: #004182;
    --li-blue-700: #0a66c2;
    --li-blue-500: #378fe9;
    --li-blue-100: #dce6f1;
    --li-green:    #057642;
    --li-red:      #b24020;
    --li-amber:    #915907;
    --li-bg:       #f3f2ee;
    --li-card:     #ffffff;
    --li-border:   #d6d0c8;
    --li-text-1:   #191919;
    --li-text-2:   #666666;
    --li-text-3:   #999999;
    --radius:      8px;
    --shadow-sm:   0 2px 8px rgba(0,0,0,.08);
    --shadow-md:   0 4px 20px rgba(0,0,0,.12);
}

/* ── Reset Streamlit chrome ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    background-color: var(--li-bg) !important;
    color: var(--li-text-1);
}
.stApp { background: var(--li-bg) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--li-card) !important;
    border-right: 1px solid var(--li-border);
}
section[data-testid="stSidebar"] * { color: var(--li-text-1) !important; }

/* ── Main container ── */
.main .block-container {
    max-width: 1040px;
    padding: 2rem 1.5rem;
}

/* ── Card component ── */
.li-card {
    background: var(--li-card);
    border: 1px solid var(--li-border);
    border-radius: var(--radius);
    padding: 24px 28px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 16px;
}

/* ── Header banner ── */
.li-header {
    background: linear-gradient(135deg, var(--li-blue-900) 0%, var(--li-blue-700) 60%, var(--li-blue-500) 100%);
    border-radius: var(--radius);
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.li-header::after {
    content: "";
    position: absolute;
    right: -60px; top: -60px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(255,255,255,.06);
}
.li-header h1 {
    color: #ffffff !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 2.1rem !important;
    font-weight: 600 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.3px;
}
.li-header p { color: rgba(255,255,255,.82) !important; font-size: .95rem; margin: 0; }

/* ── Section titles ── */
.li-section-title {
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--li-text-2);
    border-bottom: 2px solid var(--li-blue-100);
    padding-bottom: 8px;
    margin-bottom: 18px;
}

/* ── Inputs ── */
.stSelectbox > label, .stSlider > label,
.stNumberInput > label, .stRadio > label,
.stTextInput > label { 
    font-size: .82rem !important; 
    font-weight: 600 !important; 
    color: var(--li-text-2) !important; 
    letter-spacing: .3px;
}
.stSelectbox [data-baseweb="select"] > div {
    border: 1.5px solid var(--li-border) !important;
    border-radius: 4px !important;
    background: #fafaf8 !important;
}
.stSlider [data-testid="stThumb"] { background: var(--li-blue-700) !important; }
.stSlider [data-testid="stTrack"] > div { background: var(--li-blue-700) !important; }

/* ── Primary button ── */
.stButton > button {
    background: var(--li-blue-700) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 24px !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    padding: 10px 28px !important;
    letter-spacing: .2px;
    transition: background .18s, transform .12s, box-shadow .18s !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--li-blue-900) !important;
    box-shadow: 0 4px 12px rgba(10,102,194,.35) !important;
    transform: translateY(-1px);
}

/* ── Result cards ── */
.result-approved {
    background: #ebf5f0;
    border: 1.5px solid #057642;
    border-radius: var(--radius);
    padding: 28px 32px;
    text-align: center;
}
.result-rejected {
    background: #fbede8;
    border: 1.5px solid #b24020;
    border-radius: var(--radius);
    padding: 28px 32px;
    text-align: center;
}
.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-status {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.result-sub { font-size: .9rem; color: var(--li-text-2); }

/* ── Metric pill ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.metric-pill {
    background: var(--li-blue-100);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: .82rem;
    font-weight: 600;
    color: var(--li-blue-900);
}

/* ── Progress bar override ── */
.stProgress > div > div { background-color: var(--li-blue-700) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: .88rem !important;
    color: var(--li-blue-700) !important;
}

/* ── Footer ── */
.li-footer {
    text-align: center;
    font-size: .78rem;
    color: var(--li-text-3);
    padding: 20px 0 8px;
    border-top: 1px solid var(--li-border);
    margin-top: 32px;
}
.li-footer a { color: var(--li-blue-700); text-decoration: none; }

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""

st.markdown(LINKEDIN_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CATEGORICAL_MAP = {
    "education": {"Graduate": 1, "Not Graduate": 0},
    "self_employed": {"Yes": 1, "No": 0},
}


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model; fall back to a demo RandomForest if not found."""
    model_paths = [
        "model.pkl",
        "creditwise_model.pkl",
        "loan_model.pkl",
        "models/model.pkl",
        "model/loan_approval_model.pkl",
    ]
    for path in model_paths:
        if Path(path).exists():
            with open(path, "rb") as f:
                return pickle.load(f), path

    # ── Demo model (HF Spaces fallback) ──────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    st.warning(
        "⚠️  **Demo Mode** — No trained model file found. "
        "Upload `model.pkl` alongside `app.py` for real predictions.",
        icon="🔔",
    )
    X, y = make_classification(
        n_samples=1000, n_features=11, n_informative=8,
        n_redundant=2, random_state=42,
    )
    demo_model = RandomForestClassifier(n_estimators=80, random_state=42)
    demo_model.fit(X, y)
    return demo_model, "demo"


def preprocess_input(raw: dict) -> np.ndarray:
    row = {
        "no_of_dependents":        raw["no_of_dependents"],
        "education":               CATEGORICAL_MAP["education"][raw["education"]],
        "self_employed":           CATEGORICAL_MAP["self_employed"][raw["self_employed"]],
        "income_annum":            raw["income_annum"],
        "loan_amount":             raw["loan_amount"],
        "loan_term":               raw["loan_term"],
        "cibil_score":             raw["cibil_score"],
        "residential_assets_value": raw["residential_assets_value"],
        "commercial_assets_value": raw["commercial_assets_value"],
        "luxury_assets_value":     raw["luxury_assets_value"],
        "bank_asset_value":        raw["bank_asset_value"],
    }
    return np.array([list(row.values())])


def cibil_band(score: int) -> tuple:
    if score >= 750:
        return "Excellent", "🟢", "var(--li-green)"
    elif score >= 700:
        return "Good", "🟡", "var(--li-amber)"
    elif score >= 650:
        return "Fair", "🟠", "var(--li-amber)"
    else:
        return "Poor", "🔴", "var(--li-red)"


def dti_ratio(loan: float, income: float) -> float:
    if income == 0:
        return 0.0
    annual_repayment = loan / 12
    return round((annual_repayment / (income / 12)) * 100, 1)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
        <div style='font-size:1.3rem;font-weight:700;color:#0a66c2'>💼 CreditWise</div>
        <div style='font-size:.78rem;color:#666;margin-top:2px'>Loan Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 📌 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Loan Predictor", "📊 Credit Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.78rem;color:#888;line-height:1.6'>
        <strong style='color:#444'>Model Info</strong><br>
        Algorithm &nbsp;·&nbsp; Random Forest<br>
        Dataset &nbsp;·&nbsp; Loan Approval<br>
        Features &nbsp;·&nbsp; 11 inputs<br>
        Status &nbsp;·&nbsp; <span style='color:#057642;font-weight:600'>Active</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="li-header">
    <h1>💼 CreditWise Loan Intelligence</h1>
    <p>ML-powered loan approval prediction · Transparent · Fast · Reliable</p>
</div>
""", unsafe_allow_html=True)

# Load model once
model, model_src = load_model()

# ─────────────────────────────────────────────
# PAGE: LOAN PREDICTOR
# ─────────────────────────────────────────────
if "🏠 Loan Predictor" in page:

    col_form, col_result = st.columns([1.15, 1], gap="large")

    # ── LEFT: Input form ────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div class="li-section-title">👤 Applicant Profile</div>', unsafe_allow_html=True)

        education    = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
        no_of_dep    = st.slider("Number of Dependents", 0, 10, 2)
        income_annum = st.number_input(
            "Annual Income (₹)",
            min_value=100_000, max_value=10_000_000,
            value=700_000, step=50_000,
            help="Total annual income in Indian Rupees",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div class="li-section-title">🏦 Loan Details</div>', unsafe_allow_html=True)

        loan_amount = st.number_input(
            "Loan Amount Requested (₹)",
            min_value=100_000, max_value=50_000_000,
            value=2_500_000, step=100_000,
        )
        loan_term = st.slider("Loan Term (years)", 2, 30, 10)
        cibil_score = st.slider("CIBIL / Credit Score", 300, 900, 720)

        band, band_icon, _ = cibil_band(cibil_score)
        st.markdown(
            f"<div class='metric-row'>"
            f"<span class='metric-pill'>{band_icon} CIBIL: {cibil_score} — {band}</span>"
            f"<span class='metric-pill'>DTI: {dti_ratio(loan_amount, income_annum)}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div class="li-section-title">🏠 Asset Portfolio</div>', unsafe_allow_html=True)

        res_assets = st.number_input("Residential Assets Value (₹)", 0, 50_000_000, 1_500_000, 100_000)
        com_assets = st.number_input("Commercial Assets Value (₹)",  0, 50_000_000,   500_000, 100_000)
        lux_assets = st.number_input("Luxury Assets Value (₹)",      0, 30_000_000,   300_000,  50_000)
        bank_assets = st.number_input("Bank Asset / FD Value (₹)",   0, 20_000_000,   800_000, 100_000)

        total_assets = res_assets + com_assets + lux_assets + bank_assets
        ltv = round((loan_amount / total_assets * 100), 1) if total_assets > 0 else 0
        st.markdown(
            f"<div class='metric-row'>"
            f"<span class='metric-pill'>📦 Total Assets: ₹{total_assets:,.0f}</span>"
            f"<span class='metric-pill'>LTV: {ltv}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔍 Analyse Loan Eligibility", use_container_width=True)

    # ── RIGHT: Result panel ─────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="li-card" style="min-height:220px">', unsafe_allow_html=True)
        st.markdown('<div class="li-section-title">📋 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            raw = {
                "no_of_dependents":        no_of_dep,
                "education":               education,
                "self_employed":           self_employed,
                "income_annum":            income_annum,
                "loan_amount":             loan_amount,
                "loan_term":               loan_term,
                "cibil_score":             cibil_score,
                "residential_assets_value": res_assets,
                "commercial_assets_value": com_assets,
                "luxury_assets_value":     lux_assets,
                "bank_asset_value":        bank_assets,
            }
            X = preprocess_input(raw)

            try:
                prediction = model.predict(X)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    confidence = float(max(proba)) * 100
                else:
                    confidence = 85.0

                approved = int(prediction) == 1

                if approved:
                    st.markdown("""
                    <div class="result-approved">
                        <div class="result-icon">✅</div>
                        <div class="result-status" style="color:#057642">Loan Approved</div>
                        <div class="result-sub">Application meets eligibility criteria</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-rejected">
                        <div class="result-icon">❌</div>
                        <div class="result-status" style="color:#b24020">Loan Rejected</div>
                        <div class="result-sub">Application does not meet eligibility criteria</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Model Confidence**")
                st.progress(int(confidence))
                st.caption(f"{confidence:.1f}% confidence in this decision")

            except Exception as e:
                st.error(f"Prediction error: {e}")

        else:
            st.markdown("""
            <div style='text-align:center;padding:40px 20px;color:#999'>
                <div style='font-size:2.5rem'>📝</div>
                <div style='font-size:.9rem;margin-top:8px'>
                    Fill in the applicant details and click<br>
                    <strong style='color:#0a66c2'>Analyse Loan Eligibility</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Risk scorecard ─────────────────────────────────────────────
        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div class="li-section-title">⚡ Risk Scorecard</div>', unsafe_allow_html=True)

        band_label, band_icon, _ = cibil_band(cibil_score)
        dti = dti_ratio(loan_amount, income_annum)

        factors = [
            ("CIBIL Score", cibil_score, 900, f"{band_icon} {band_label}"),
            ("Income Sufficiency",
             min(income_annum / 10_000_000 * 100, 100), 100,
             f"₹{income_annum/100_000:.1f}L/yr"),
            ("Asset Coverage",
             min(total_assets / loan_amount * 100, 100) if loan_amount else 0, 100,
             f"LTV {ltv}%"),
            ("Low DTI Ratio",
             max(0, 100 - dti), 100,
             f"DTI {dti}%"),
        ]

        for label, val, max_val, caption in factors:
            pct = int((val / max_val) * 100)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:.82rem;margin-bottom:2px'>"
                f"<span style='font-weight:600'>{label}</span>"
                f"<span style='color:#666'>{caption}</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(pct)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Tips ───────────────────────────────────────────────────────
        with st.expander("💡 Tips to improve eligibility"):
            st.markdown("""
- **Raise your CIBIL score** above 750 by clearing pending dues  
- **Reduce loan amount** or extend the term to lower EMI burden  
- **Increase declared assets** — include FDs, NSC, PPF where applicable  
- **Pay off existing loans** to lower your Debt-to-Income ratio  
- **Graduate status** marginally improves approval odds in this model  
            """)

# ─────────────────────────────────────────────
# PAGE: CREDIT INSIGHTS
# ─────────────────────────────────────────────
elif "📊 Credit Insights" in page:
    st.markdown('<div class="li-card">', unsafe_allow_html=True)
    st.markdown('<div class="li-section-title">📊 Credit Score Reference Chart</div>', unsafe_allow_html=True)

    cibil_data = pd.DataFrame({
        "Band":       ["Excellent (750–900)", "Good (700–749)", "Fair (650–699)", "Poor (300–649)"],
        "Min Score":  [750, 700, 650, 300],
        "Max Score":  [900, 749, 699, 649],
        "Approval %": [91, 72, 48, 18],
        "Avg Interest Rate (%)": [7.5, 9.2, 11.8, 15.0],
    })

    st.dataframe(
        cibil_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Approval %": st.column_config.ProgressColumn(
                "Approval Rate", min_value=0, max_value=100, format="%d%%"
            )
        },
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="li-card">', unsafe_allow_html=True)
    st.markdown('<div class="li-section-title">🔑 Key Approval Factors</div>', unsafe_allow_html=True)

    facts = [
        ("🎯", "CIBIL Score",       "Most critical factor. Score ≥ 750 dramatically increases approval odds."),
        ("💰", "Income vs Loan",    "DTI below 40% is ideal. Banks prefer EMI ≤ 40% of monthly income."),
        ("🏠", "Asset Coverage",    "Total assets ≥ 1.5× loan amount signals low risk to lenders."),
        ("📋", "Employment Status", "Salaried applicants have higher baseline approval vs self-employed."),
        ("👨‍👩‍👧", "Dependents",       "More dependents slightly reduce approval probability due to higher living costs."),
    ]
    for icon, title, desc in facts:
        st.markdown(
            f"<div style='display:flex;gap:14px;margin-bottom:14px;align-items:flex-start'>"
            f"<div style='font-size:1.4rem'>{icon}</div>"
            f"<div><div style='font-weight:600;font-size:.9rem'>{title}</div>"
            f"<div style='font-size:.82rem;color:#666;margin-top:2px'>{desc}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif "ℹ️ About" in page:
    st.markdown('<div class="li-card">', unsafe_allow_html=True)
    st.markdown('<div class="li-section-title">ℹ️ About CreditWise</div>', unsafe_allow_html=True)
    st.markdown("""
**CreditWise** is an ML-powered Loan Approval Prediction System built as part of the 
**AIML practitioner portfolio** at Anna University Regional Campus, Tirunelveli.

#### 🛠 Tech Stack
| Layer | Technology |
|-------|------------|
| UI / Frontend | Streamlit |
| ML Model | Scikit-learn (Random Forest / XGBoost) |
| Data Processing | Pandas, NumPy |
| Deployment | Hugging Face Spaces |
| Version Control | Git + GitHub |

#### 📦 Dataset
- **Source**: Loan Approval Prediction Dataset (Kaggle)
- **Features**: 11 applicant & financial features
- **Target**: Loan Status — Approved / Rejected

#### 🎓 Author
**Karthika Krishna M** — CSE, Anna University Regional Campus Tirunelveli  
_"Building intelligent systems that solve real financial problems."_
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="li-footer">
    LendSure-AI · Built by <strong>Karthika Krishna M</strong> · 
    <a href="https://github.com/KARTHIKAKRISHNA123" target="_blank">GitHub</a> · 
    
    <em>For educational & demonstration purposes only</em>
</div>
""", unsafe_allow_html=True)