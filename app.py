"""
LendSure AI — Loan Approval Intelligence Platform
app.py  |  Hugging Face Spaces (Streamlit SDK)
Author : Karthika Krishna M | github.com/KARTHIKAKRISHNA123

Exact pipeline from Credit_Wise_Loan_Approval_System.ipynb:
  Cell 12  → Impute numerics (mean)
  Cell 14  → Impute categoricals (most_frequent)
  Cell 41  → Drop Applicant_ID
  Cell 46  → LabelEncode: Education_Level, Loan_Approved
  Cell 48  → OHE (drop=first): Employment_Status, Marital_Status,
             Loan_Purpose, Property_Area, Gender, Employer_Category
  Cell 80  → DTI_Ratio_sq = DTI_Ratio**2
             Credit_Score_sq = Credit_Score**2
             Drop: Credit_Score, DTI_Ratio
             (Applicant_Income_log is COMMENTED OUT in notebook)
  Cell 65  → StandardScaler

Final feature set: 27 columns (verified by retraining)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="LendSure AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navy-blue accent on HF default theme — no split divs ─────────────────────
st.markdown("""
<style>
:root {
  --navy:   #1B3A6B;
  --navym:  #2563A8;
  --navyl:  #EBF2FF;
  --grn:    #166534;
  --grnbg:  #F0FDF4;
  --red:    #991B1B;
  --redbg:  #FEF2F2;
}
.ls-header {
  background: linear-gradient(120deg,#1B3A6B 0%,#2563A8 60%,#3B82F6 100%);
  border-radius:10px; padding:28px 32px; margin-bottom:20px;
}
.ls-header h1 {
  color:#fff !important; font-size:1.9rem !important;
  font-weight:700 !important; margin:0 0 6px !important;
}
.ls-header p { color:rgba(255,255,255,.85) !important; font-size:.88rem; margin:0; }
.ls-sec {
  font-size:.72rem; font-weight:700; letter-spacing:1px;
  text-transform:uppercase; color:var(--navym);
  border-bottom:2px solid var(--navyl);
  padding-bottom:6px; margin-bottom:14px; margin-top:4px;
}
.res-ok {
  background:var(--grnbg); border:2px solid #16A34A;
  border-radius:10px; padding:24px 16px;
  text-align:center; margin-bottom:16px;
}
.res-no {
  background:var(--redbg); border:2px solid #DC2626;
  border-radius:10px; padding:24px 16px;
  text-align:center; margin-bottom:16px;
}
.res-wait {
  border:2px dashed #CBD5E1; border-radius:10px;
  padding:36px 16px; text-align:center;
  margin-bottom:16px; color:#94A3B8;
}
.res-ico   { font-size:2.5rem; line-height:1; margin-bottom:8px; }
.res-title { font-size:1.25rem; font-weight:700; margin-bottom:4px; }
.res-sub   { font-size:.83rem; opacity:.8; }
.pill {
  display:inline-block; background:var(--navyl); color:var(--navy);
  border-radius:20px; padding:3px 11px;
  font-size:.75rem; font-weight:600; margin:2px 3px 4px 0;
}
.pill-m {
  display:inline-block; background:var(--navy); color:#fff;
  border-radius:20px; padding:4px 14px;
  font-size:.76rem; font-weight:600; margin-bottom:10px;
}
.sc-row {
  display:flex; justify-content:space-between;
  font-size:.8rem; margin-bottom:2px;
}
.sc-lbl { font-weight:600; }
.sc-val { color:#64748B; }
.ins-row { display:flex; gap:12px; margin-bottom:14px; align-items:flex-start; }
.ins-ico  { font-size:1.3rem; flex-shrink:0; }
.ins-ttl  { font-weight:600; font-size:.87rem; }
.ins-dsc  { font-size:.79rem; color:#64748B; margin-top:2px; }
.ls-foot {
  text-align:center; font-size:.74rem; color:#94A3B8;
  border-top:1px solid #E2E8F0; padding-top:16px; margin-top:24px;
}
.ls-foot a { color:var(--navym); text-decoration:none; }
.stProgress>div>div { background:var(--navym) !important; }
.stButton>button {
  background:var(--navy) !important; color:#fff !important;
  border:none !important; border-radius:8px !important;
  font-weight:600 !important; width:100%; transition:background .15s !important;
}
.stButton>button:hover { background:var(--navym) !important; }
#MainMenu,footer,header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — exact OHE column order from notebook Cell 48
# ─────────────────────────────────────────────────────────────────────────────
OHE_COLS = ["Employment_Status","Marital_Status","Loan_Purpose",
            "Property_Area","Gender","Employer_Category"]

# UI options — from CSV unique values (post-imputation)
EMPLOYMENT_OPTS = ["Salaried","Self-employed","Contract","Unemployed"]
MARITAL_OPTS    = ["Married","Single"]
PURPOSE_OPTS    = ["Personal","Car","Business","Home","Education"]
PROPERTY_OPTS   = ["Urban","Semiurban","Rural"]
GENDER_OPTS     = ["Male","Female"]
EMPLOYER_OPTS   = ["Private","Government","MNC","Business","Unemployed"]
EDUCATION_OPTS  = ["Graduate","Not Graduate"]

MODEL_OPTIONS = {
    "⭐ Logistic Regression (Best)": "model_lr.pkl",
    "K-Nearest Neighbours (KNN)":    "model_knn.pkl",
    "Naive Bayes (Best Precision)":  "model_nb.pkl",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_scaler():
    p = Path("scaler.pkl")
    return pickle.load(open(p,"rb")) if p.exists() else None

@st.cache_resource(show_spinner=False)
def load_encoder():
    p = Path("encoder.pkl")
    return pickle.load(open(p,"rb")) if p.exists() else None

@st.cache_resource(show_spinner=False)
def load_model(name:str):
    p = Path(name)
    return pickle.load(open(p,"rb")) if p.exists() else None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — verified against 27-column feature set
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(raw:dict, scaler, encoder) -> np.ndarray:
    # LabelEncode Education_Level (Graduate=0, Not Graduate=1)
    edu_enc = 1 if raw["Education_Level"] == "Not Graduate" else 0

    # Numerical features
    num = {
        "Applicant_Income":   raw["Applicant_Income"],
        "Coapplicant_Income": raw["Coapplicant_Income"],
        "Age":                raw["Age"],
        "Dependents":         raw["Dependents"],
        "Existing_Loans":     raw["Existing_Loans"],
        "Savings":            raw["Savings"],
        "Collateral_Value":   raw["Collateral_Value"],
        "Loan_Amount":        raw["Loan_Amount"],
        "Loan_Term":          raw["Loan_Term"],
        "Education_Level":    edu_enc,
        # Engineered features (Cell 80) — Credit_Score & DTI_Ratio originals dropped
        "DTI_Ratio_sq":       raw["DTI_Ratio"] ** 2,
        "Credit_Score_sq":    raw["Credit_Score"] ** 2,
    }

    # OHE categorical block
    cat_df = pd.DataFrame([{c: raw[c] for c in OHE_COLS}])
    if encoder is not None:
        ohe_arr = encoder.transform(cat_df)
        ohe_df  = pd.DataFrame(ohe_arr, columns=encoder.get_feature_names_out(OHE_COLS))
    else:
        ohe_df = pd.get_dummies(cat_df, drop_first=True)

    num_df  = pd.DataFrame([num])
    full_df = pd.concat([num_df.reset_index(drop=True),
                         ohe_df.reset_index(drop=True)], axis=1)

    # ── CRITICAL: reindex to EXACT column order scaler was fitted on ──────────
    # Verified order from scaler.feature_names_in_ (27 columns):
    # [0-9]  numerics + Education_Level
    # [10-24] OHE columns in encoder.get_feature_names_out() order
    # [25-26] DTI_Ratio_sq, Credit_Score_sq
    EXACT_COLS = (
        ["Applicant_Income","Coapplicant_Income","Age","Dependents",
         "Existing_Loans","Savings","Collateral_Value","Loan_Amount",
         "Loan_Term","Education_Level"]
        + list(encoder.get_feature_names_out(OHE_COLS))
        + ["DTI_Ratio_sq","Credit_Score_sq"]
    ) if encoder is not None else list(full_df.columns)

    full_df = full_df.reindex(columns=EXACT_COLS, fill_value=0)

    if scaler is not None:
        return scaler.transform(full_df)
    return full_df.values

def credit_band(score:int):
    if score >= 750: return "Excellent","🟢"
    if score >= 700: return "Good","🟡"
    if score >= 650: return "Fair","🟠"
    return "Poor","🔴"

def fmt(v:float)->str:
    if v>=1e7: return f"₹{v/1e7:.1f}Cr"
    if v>=1e5: return f"₹{v/1e5:.1f}L"
    return f"₹{v:,.0f}"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 LendSure AI")
    st.caption("Loan Intelligence Platform")
    st.divider()

    page = st.radio("Navigate",
                    ["🏠  Predict","📊  Insights","🤖  Models","ℹ️  About"])
    st.divider()

    st.markdown("**Select Model**")
    sel_label = st.radio("Model", list(MODEL_OPTIONS.keys()),
                         label_visibility="collapsed")
    sel_pkl   = MODEL_OPTIONS[sel_label]
    short     = sel_label.split("(")[0].strip()
    st.divider()

    st.markdown(f"""
**Active:** {short}  
**Dataset:** 1,000 records · 19 features  
**Features used:** 27 (post-engineering)  
**Models:** LR · KNN · Naive Bayes  
**Status:** 🟢 Running
""")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='ls-header'>"
    "<h1>🏦 LendSure AI</h1>"
    "<p>ML-powered loan approval · Logistic Regression · KNN · Naive Bayes · Transparent · Fast</p>"
    "</div>",
    unsafe_allow_html=True,
)

scaler  = load_scaler()
encoder = load_encoder()
missing = [n for n,o in [("scaler.pkl",scaler),("encoder.pkl",encoder)] if o is None]
if missing:
    st.warning(f"⚠️ {' and '.join(missing)} not found — demo mode. "
               "Upload all pkl files for real predictions.", icon="🔔")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if "Predict" in page:
    col_l, col_r = st.columns([1.05,1], gap="large")

    with col_l:
        # Section 1 — Applicant Profile
        st.markdown("<div class='ls-sec'>👤 Applicant Profile</div>",
                    unsafe_allow_html=True)
        a1,a2 = st.columns(2)
        with a1:
            age    = st.number_input("Age", 18, 75, 35)
            gender = st.selectbox("Gender", GENDER_OPTS)
            marital= st.selectbox("Marital Status", MARITAL_OPTS)
        with a2:
            dep    = st.number_input("Dependents", 0, 10, 1)
            edu    = st.selectbox("Education Level", EDUCATION_OPTS)
            emp_st = st.selectbox("Employment Status", EMPLOYMENT_OPTS)
        emp_cat = st.selectbox("Employer Category", EMPLOYER_OPTS)
        st.divider()

        # Section 2 — Financials
        st.markdown("<div class='ls-sec'>💰 Financial Details</div>",
                    unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        with b1:
            app_inc   = st.number_input("Applicant Income (₹/mo)",    0, 200_000, 8_000, 500)
            coapp_inc = st.number_input("Co-applicant Income (₹/mo)", 0, 100_000, 2_000, 500)
            savings   = st.number_input("Savings Balance (₹)",        0, 500_000,15_000,1_000)
        with b2:
            cr_score = st.slider("Credit Score", 300, 900, 680)
            ex_loans = st.number_input("Existing Loans (#)", 0, 10, 1)
            dti      = st.slider("DTI Ratio", 0.0, 1.0, 0.35, 0.01,
                                 help="Debt-to-Income ratio (0=no debt, 1=all income goes to debt)")
        band_lbl, band_ico = credit_band(cr_score)
        st.markdown(
            f"<span class='pill'>{band_ico} Credit: {cr_score} — {band_lbl}</span>"
            f"<span class='pill'>DTI: {dti:.0%}</span>"
            f"<span class='pill'>Existing Loans: {int(ex_loans)}</span>",
            unsafe_allow_html=True)
        st.divider()

        # Section 3 — Loan & Property
        st.markdown("<div class='ls-sec'>🏦 Loan & Property Details</div>",
                    unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            loan_amt  = st.number_input("Loan Amount (₹)",     5_000,1_000_000,30_000,1_000)
            loan_term = st.number_input("Loan Term (months)",     12,      360,    84,   12)
            loan_purp = st.selectbox("Loan Purpose", PURPOSE_OPTS)
        with c2:
            collateral = st.number_input("Collateral Value (₹)", 0, 2_000_000, 50_000, 5_000)
            prop_area  = st.selectbox("Property Area", PROPERTY_OPTS)

        ltv = round(loan_amt/collateral*100,1) if collateral>0 else 0
        emi = round(loan_amt/loan_term,0)      if loan_term>0  else 0
        st.markdown(
            f"<span class='pill'>LTV: {ltv}%</span>"
            f"<span class='pill'>Est. EMI: {fmt(emi)}/mo</span>"
            f"<span class='pill'>Term: {int(loan_term)} mo</span>",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<span class='pill-m'>Model: {short}</span>",
                    unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Loan Eligibility", use_container_width=True)

    with col_r:
        # Result
        st.markdown("<div class='ls-sec'>📋 Prediction Result</div>",
                    unsafe_allow_html=True)

        if predict_btn:
            raw = {
                "Applicant_Income":   app_inc,
                "Coapplicant_Income": coapp_inc,
                "Employment_Status":  emp_st,
                "Age":                age,
                "Marital_Status":     marital,
                "Dependents":         dep,
                "Credit_Score":       cr_score,
                "Existing_Loans":     ex_loans,
                "DTI_Ratio":          dti,
                "Savings":            savings,
                "Collateral_Value":   collateral,
                "Loan_Amount":        loan_amt,
                "Loan_Term":          loan_term,
                "Loan_Purpose":       loan_purp,
                "Property_Area":      prop_area,
                "Education_Level":    edu,
                "Gender":             gender,
                "Employer_Category":  emp_cat,
            }
            model = load_model(sel_pkl)
            if model is None:
                st.error(f"**{sel_pkl}** not found. Upload pkl files via the Files tab.")
            else:
                try:
                    X        = preprocess(raw, scaler, encoder)
                    pred     = model.predict(X)[0]
                    approved = int(pred) == 1
                    conf     = (float(max(model.predict_proba(X)[0]))*100
                                if hasattr(model,"predict_proba") else 75.0)

                    if approved:
                        st.markdown(
                            "<div class='res-ok'>"
                            "<div class='res-ico'>✅</div>"
                            "<div class='res-title' style='color:#166534'>Loan Approved</div>"
                            "<div class='res-sub'>Application meets eligibility criteria</div>"
                            "</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<div class='res-no'>"
                            "<div class='res-ico'>❌</div>"
                            "<div class='res-title' style='color:#991B1B'>Loan Rejected</div>"
                            "<div class='res-sub'>Application does not meet eligibility criteria</div>"
                            "</div>", unsafe_allow_html=True)

                    st.markdown(f"**Model Confidence** — `{conf:.1f}%`")
                    st.progress(min(int(conf),100))

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.caption("Tip: Re-upload scaler.pkl and encoder.pkl from the retrained artifacts.")
        else:
            st.markdown(
                "<div class='res-wait'>"
                "<div class='res-ico'>📝</div>"
                "<div style='font-size:.86rem;margin-top:8px'>"
                "Fill in applicant details on the left<br>and click "
                "<b style='color:#2563A8'>Predict Loan Eligibility</b>"
                "</div></div>", unsafe_allow_html=True)

        st.divider()

        # Live Risk Scorecard
        st.markdown("<div class='ls-sec'>⚡ Live Risk Scorecard</div>",
                    unsafe_allow_html=True)
        cr_pct   = int((cr_score-300)/600*100)
        inc_pct  = min(int(app_inc/200_000*100),100)
        sav_pct  = min(int(savings/500_000*100),100)
        dti_pct  = max(0,int((1-dti)*100))
        coll_pct = min(int(collateral/loan_amt*100) if loan_amt else 0,100)

        for lbl,pct,cap in [
            ("Credit Score",   cr_pct,   f"{cr_score} — {band_lbl}"),
            ("Income Level",   inc_pct,  fmt(app_inc)+"/mo"),
            ("Savings Buffer", sav_pct,  fmt(savings)),
            ("Low DTI",        dti_pct,  f"{dti:.0%} DTI"),
            ("Collateral",     coll_pct, f"LTV {ltv}%"),
        ]:
            st.markdown(
                f"<div class='sc-row'><span class='sc-lbl'>{lbl}</span>"
                f"<span class='sc-val'>{cap}</span></div>",
                unsafe_allow_html=True)
            st.progress(pct)

        with st.expander("💡 Tips to improve eligibility"):
            st.markdown("""
- **Credit Score ≥ 700** dramatically improves approval odds
- **DTI Ratio < 0.40** — clear existing loans before applying
- **Collateral > Loan Amount** reduces lender risk
- **Savings ≥ 3× EMI** signals financial stability
- **Salaried + Government employer** scores highest
- **Urban property area** has better approval rates
            """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif "Insights" in page:
    st.markdown("<div class='ls-sec'>📊 Credit Score Reference</div>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Band":           ["Excellent (750–900)","Good (700–749)","Fair (650–699)","Poor (300–649)"],
        "Approval Rate %":[91,72,48,18],
        "Avg Interest %": [8.5,10.2,12.5,16.0],
        "Risk Level":     ["Very Low","Low","Medium","High"],
    }), use_container_width=True, hide_index=True,
    column_config={"Approval Rate %": st.column_config.ProgressColumn(
        "Approval Rate", min_value=0, max_value=100, format="%d%%")})

    st.divider()
    st.markdown("<div class='ls-sec'>📈 Dataset Overview</div>",
                unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Records","1,000")
    m2.metric("Raw Features","19")
    m3.metric("Approved","29.8%")
    m4.metric("Rejected","65.2%")
    st.caption("Class-imbalanced — F1 Score and Precision are primary metrics, not just accuracy.")

    st.divider()
    st.markdown("<div class='ls-sec'>🔑 Key Approval Factors (from EDA)</div>",
                unsafe_allow_html=True)
    for ico,ttl,dsc in [
        ("🎯","Credit Score","Strongest predictor. Score ≥ 700 significantly boosts approval."),
        ("💰","DTI Ratio","Debt-to-income ratio. Below 0.40 strongly preferred by lenders."),
        ("🏦","Collateral Value","Higher collateral → lower lender risk → better approval odds."),
        ("💼","Employment Status","Salaried > Self-employed > Contract. Unemployed rarely approved."),
        ("🏠","Property Area","Urban > Semiurban > Rural in approval likelihood."),
        ("💵","Savings Balance","Higher savings signal resilience and repayment capacity."),
        ("👨‍👩‍👧","Dependents","More dependents reduce disposable income — moderate negative impact."),
    ]:
        st.markdown(
            f"<div class='ins-row'><div class='ins-ico'>{ico}</div>"
            f"<div><div class='ins-ttl'>{ttl}</div>"
            f"<div class='ins-dsc'>{dsc}</div></div></div>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif "Models" in page:
    st.markdown("<div class='ls-sec'>🤖 Model Comparison</div>",
                unsafe_allow_html=True)
    st.caption("All models share the same pipeline — same scaler, same encoder, same 27 features.")
    st.dataframe(pd.DataFrame({
        "Model":     ["Logistic Regression ⭐","K-Nearest Neighbours","Naive Bayes"],
        "File":      ["model_lr.pkl","model_knn.pkl","model_nb.pkl"],
        "Accuracy":  ["~87.5%","~75.5%","~86.5%"],
        "Precision": ["~79%","~62%","~78%"],
        "F1 Score":  ["~80%","~56%","~78%"],
        "Best For":  ["Default / overall","Pattern similarity","Minimising false approvals"],
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("<div class='ls-sec'>🔧 Preprocessing Pipeline (exact notebook)</div>",
                unsafe_allow_html=True)
    st.code(
        "Raw CSV  (1000 rows × 20 cols)\n"
        "  │\n"
        "  ├─ Impute: numerics=mean | categoricals=most_frequent\n"
        "  ├─ Drop: Applicant_ID\n"
        "  ├─ LabelEncode: Education_Level, Loan_Approved\n"
        "  ├─ OneHotEncode (drop=first):\n"
        "  │     Employment_Status, Marital_Status, Loan_Purpose,\n"
        "  │     Property_Area, Gender, Employer_Category\n"
        "  ├─ Feature Engineering (Cell 80):\n"
        "  │     DTI_Ratio_sq    = DTI_Ratio ** 2\n"
        "  │     Credit_Score_sq = Credit_Score ** 2\n"
        "  │     # Applicant_Income_log → COMMENTED OUT (not used)\n"
        "  ├─ Drop originals: Credit_Score, DTI_Ratio\n"
        "  └─ StandardScaler → 27 final features\n",
        language="text")

    st.divider()
    st.markdown("<div class='ls-sec'>📦 Artifact Status</div>",
                unsafe_allow_html=True)
    for fname,desc in [
        ("model_lr.pkl",  "LogisticRegression() — Acc 87.5% · Prec 79%"),
        ("model_knn.pkl", "KNeighborsClassifier(n_neighbors=5) — Acc 75.5%"),
        ("model_nb.pkl",  "GaussianNB() — Acc 86.5% · Best Precision"),
        ("scaler.pkl",    "StandardScaler fitted on 27-feature X_train"),
        ("encoder.pkl",   "OneHotEncoder(drop='first', sparse_output=False)"),
    ]:
        a,b,c = st.columns([2,3,1])
        a.code(fname)
        b.caption(desc)
        c.markdown("✅ Loaded" if Path(fname).exists() else "⚠️ Missing")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif "About" in page:
    st.markdown("<div class='ls-sec'>ℹ️ About LendSure AI</div>",
                unsafe_allow_html=True)
    st.markdown("""
**LendSure AI** is an end-to-end ML loan approval prediction system built as part of the
AIML practitioner portfolio at **Anna University Regional Campus, Tirunelveli**.

Three classifiers are trained on 1,000 real loan records through a full preprocessing pipeline
and served via a transparent Streamlit UI with live risk scoring and model switching.

| Layer | Technology |
|-------|------------|
| UI | Streamlit (HF Spaces) |
| ML Models | Scikit-learn — LR · KNN · Naive Bayes |
| Preprocessing | SimpleImputer + LabelEncoder + OHE + StandardScaler |
| Feature Eng. | DTI² · Credit Score² |
| Final Features | 27 columns (verified by retraining) |
| Dataset | loan_approval_data.csv — 1,000 rows · 20 raw cols |
| Version Control | Git LFS (pkl files) + GitHub |
""")
    st.divider()
    st.markdown("""
**Author:** Karthika Krishna M — CSE, Anna University Regional Campus, Tirunelveli  
🔗 [github.com/KARTHIKAKRISHNA123](https://github.com/KARTHIKAKRISHNA123) ·
🤗 [KarthikaKrishna123 on Hugging Face](https://huggingface.co/KarthikaKrishna123)
""")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='ls-foot'>LendSure AI · Built by <strong>Karthika Krishna M</strong> · "
    "<a href='https://github.com/KARTHIKAKRISHNA123' target='_blank'>GitHub</a> · "
    "Anna University Regional Campus, Tirunelveli &nbsp;|&nbsp;"
    "<em>For educational &amp; demonstration purposes only</em></div>",
    unsafe_allow_html=True)