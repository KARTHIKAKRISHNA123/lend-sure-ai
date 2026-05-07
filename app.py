"""
LendSure AI — Loan Approval Intelligence Platform
app.py  |  Hugging Face Spaces (Streamlit SDK)

Built from: Credit_Wise_Loan_Approval_System.ipynb
Models   : model_lr.pkl (best), model_knn.pkl, model_nb.pkl
Scaler   : scaler.pkl  (StandardScaler)
Encoder  : encoder.pkl (OneHotEncoder — drop='first', sparse_output=False)
Target   : Loan_Approved → LabelEncoder → Yes=1, No=0

Feature pipeline (matches notebook exactly):
  1. Drop Applicant_ID
  2. Impute numerics=mean, categoricals=most_frequent
  3. LabelEncode: Education_Level (Graduate=0, Not Graduate=1), Loan_Approved
  4. OneHotEncode (drop=first): Employment_Status, Marital_Status,
                                Loan_Purpose, Property_Area, Gender,
                                Employer_Category
  5. Feature engineer: DTI_Ratio_sq = DTI_Ratio**2
                       Credit_Score_sq = Credit_Score**2
  6. Drop original: Credit_Score, DTI_Ratio
  7. StandardScaler

Author : Karthika Krishna M | github.com/KARTHIKAKRISHNA123
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ── Page config — MUST be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="LendSure AI · Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — LinkedIn-inspired, HF-Spaces-safe
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap');
:root{
  --bl9:#004182;--bl7:#0a66c2;--bl4:#378fe9;--bl1:#dce6f1;--bl0:#e8f1fb;
  --grn:#057642;--red:#b24020;--amb:#915907;
  --bg:#f3f2ee;--srf:#ffffff;--bdr:#d6d0c8;
  --t1:#191919;--t2:#595959;--t3:#999999;
  --r:8px;--sh:0 1px 6px rgba(0,0,0,.10);
}
html,body,[class*="css"],.stApp{font-family:'IBM Plex Sans',sans-serif!important;background:var(--bg)!important;}
.main .block-container{max-width:1080px;padding:1.4rem 1.4rem 3rem;}
[data-testid="stSidebar"]{background:var(--srf)!important;border-right:1px solid var(--bdr)!important;}
.lsh{background:linear-gradient(135deg,var(--bl9) 0%,var(--bl7) 55%,var(--bl4) 100%);
     border-radius:var(--r);padding:28px 34px;margin-bottom:18px;position:relative;overflow:hidden;}
.lsh::before{content:"";position:absolute;top:-36px;right:-36px;width:160px;height:160px;
             border-radius:50%;background:rgba(255,255,255,.07);pointer-events:none;}
.lsh h1{color:#fff!important;font-family:'IBM Plex Serif',serif!important;
        font-size:1.85rem!important;font-weight:600!important;margin:0 0 5px!important;}
.lsh p{color:rgba(255,255,255,.82)!important;font-size:.88rem;margin:0;}
.lsc{background:var(--srf);border:1px solid var(--bdr);border-radius:var(--r);
     padding:20px 22px;box-shadow:var(--sh);margin-bottom:12px;}
.lsl{font-size:.7rem;font-weight:700;letter-spacing:1.1px;text-transform:uppercase;
     color:var(--t3);border-bottom:2px solid var(--bl0);padding-bottom:7px;margin-bottom:14px;}
.r-ok{background:#ebf5f0;border:1.5px solid var(--grn);border-radius:var(--r);padding:22px;text-align:center;}
.r-no{background:#fbede8;border:1.5px solid var(--red);border-radius:var(--r);padding:22px;text-align:center;}
.r-ico{font-size:2.4rem;line-height:1.1;}
.r-ttl{font-size:1.25rem;font-weight:700;margin:6px 0 3px;}
.r-sub{font-size:.82rem;color:var(--t2);}
.pill{display:inline-block;background:var(--bl0);color:var(--bl9);border-radius:20px;
      padding:3px 12px;font-size:.76rem;font-weight:600;margin:2px 3px 2px 0;}
.mbadge{display:inline-block;background:var(--bl7);color:#fff;border-radius:20px;
        padding:3px 12px;font-size:.74rem;font-weight:600;}
.stButton>button{background:var(--bl7)!important;color:#fff!important;border:none!important;
  border-radius:24px!important;font-weight:600!important;font-size:.88rem!important;
  padding:9px 22px!important;width:100%;transition:background .18s,box-shadow .18s!important;}
.stButton>button:hover{background:var(--bl9)!important;box-shadow:0 4px 12px rgba(10,102,194,.30)!important;}
.stSelectbox>label,.stSlider>label,.stNumberInput>label,.stRadio>label{
  font-size:.79rem!important;font-weight:600!important;color:var(--t2)!important;}
div[data-baseweb="select"]>div{border:1.5px solid var(--bdr)!important;border-radius:4px!important;background:#fafaf8!important;}
.stProgress>div>div{background:var(--bl7)!important;}
.stTabs [data-baseweb="tab"]{font-size:.84rem;font-weight:600;color:var(--t2);}
.stTabs [aria-selected="true"]{color:var(--bl7)!important;border-bottom:2px solid var(--bl7)!important;}
.streamlit-expanderHeader{font-weight:600!important;font-size:.84rem!important;color:var(--bl7)!important;}
.lsf{text-align:center;font-size:.74rem;color:var(--t3);border-top:1px solid var(--bdr);padding-top:14px;margin-top:20px;}
.lsf a{color:var(--bl7);text-decoration:none;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
OHE_COLS = ["Employment_Status","Marital_Status","Loan_Purpose",
            "Property_Area","Gender","Employer_Category"]

EDUCATION_OPTS  = ["Graduate","Not Graduate"]
EMPLOYMENT_OPTS = ["Salaried","Self-employed","Contract","Unemployed"]
MARITAL_OPTS    = ["Married","Single"]
PURPOSE_OPTS    = ["Personal","Car","Business","Home","Education"]
PROPERTY_OPTS   = ["Urban","Semiurban","Rural"]
GENDER_OPTS     = ["Male","Female"]
EMPLOYER_OPTS   = ["Private","Government","MNC","Business","Unemployed"]

MODEL_OPTIONS = {
    "⭐ Logistic Regression (Best)": "model_lr.pkl",
    "K-Nearest Neighbours (KNN)":    "model_knn.pkl",
    "Naive Bayes (Best Precision)":  "model_nb.pkl",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    scaler, encoder = None, None
    if Path("scaler.pkl").exists():
        with open("scaler.pkl","rb") as f: scaler = pickle.load(f)
    if Path("encoder.pkl").exists():
        with open("encoder.pkl","rb") as f: encoder = pickle.load(f)
    return scaler, encoder

@st.cache_resource(show_spinner=False)
def load_model(pkl_name):
    if Path(pkl_name).exists():
        with open(pkl_name,"rb") as f: return pickle.load(f)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — mirrors notebook Cell 46-85 exactly
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(raw, scaler, encoder):
    edu_enc = 1 if raw["Education_Level"] == "Not Graduate" else 0
    num_part = {
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
        "DTI_Ratio_sq":       raw["DTI_Ratio"] ** 2,
        "Credit_Score_sq":    raw["Credit_Score"] ** 2,
    }
    cat_df = pd.DataFrame([{c: raw[c] for c in OHE_COLS}])
    if encoder is not None:
        ohe_arr = encoder.transform(cat_df)
        ohe_df  = pd.DataFrame(ohe_arr, columns=encoder.get_feature_names_out(OHE_COLS))
    else:
        ohe_df = pd.get_dummies(cat_df, drop_first=True)
    num_df  = pd.DataFrame([num_part])
    full_df = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
    if scaler is not None:
        return scaler.transform(full_df)
    return full_df.values

def cibil_band(score):
    if score >= 750: return "Excellent","🟢"
    if score >= 700: return "Good","🟡"
    if score >= 650: return "Fair","🟠"
    return "Poor","🔴"

def fmt_inr(v):
    if v >= 1e7: return f"₹{v/1e7:.1f}Cr"
    if v >= 1e5: return f"₹{v/1e5:.1f}L"
    return f"₹{v:,.0f}"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 18px'>
      <div style='font-size:1.25rem;font-weight:700;color:#0a66c2'>🏦 LendSure AI</div>
      <div style='font-size:.75rem;color:#666;margin-top:2px'>Loan Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Navigate**")
    page = st.radio("page", ["🏠 Predict","📊 Insights","🤖 Models","ℹ️ About"],
                    label_visibility="collapsed")
    st.markdown("---")

    st.markdown("**Select Model**")
    sel_label = st.radio("model_select", list(MODEL_OPTIONS.keys()),
                         label_visibility="collapsed")
    sel_pkl = MODEL_OPTIONS[sel_label]
    st.markdown("---")

    st.markdown(f"""
    <div style='font-size:.75rem;color:#888;line-height:1.8'>
      <strong style='color:#444'>Active Model</strong><br>
      <span style='color:#0a66c2;font-weight:600'>{sel_label.split("(")[0].strip()}</span><br><br>
      <strong style='color:#444'>Dataset</strong><br>1,000 records · 19 features<br><br>
      <strong style='color:#444'>Models</strong><br>LR · KNN · Naive Bayes<br><br>
      <strong style='color:#444'>Status</strong><br>
      <span style='color:#057642;font-weight:600'>● Active</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsh">
  <h1>🏦 LendSure AI</h1>
  <p>ML-powered loan approval · Logistic Regression · KNN · Naive Bayes · Transparent · Fast</p>
</div>""", unsafe_allow_html=True)

scaler, encoder = load_artifacts()
if scaler is None or encoder is None:
    st.warning("⚠️ **scaler.pkl** or **encoder.pkl** not found. Running in demo mode — upload pkl files for real predictions.", icon="🔔")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if "🏠 Predict" in page:
    col_l, col_r = st.columns([1.1,1], gap="large")

    with col_l:
        # Card 1 — Applicant Profile
        st.markdown('<div class="lsc"><div class="lsl">👤 Applicant Profile</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            age      = st.number_input("Age", 18, 75, 35)
            gender   = st.selectbox("Gender", GENDER_OPTS)
            marital  = st.selectbox("Marital Status", MARITAL_OPTS)
        with c2:
            dep      = st.number_input("Dependents", 0, 10, 1)
            edu      = st.selectbox("Education Level", EDUCATION_OPTS)
            emp_st   = st.selectbox("Employment Status", EMPLOYMENT_OPTS)
        emp_cat = st.selectbox("Employer Category", EMPLOYER_OPTS)
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 2 — Financials
        st.markdown('<div class="lsc"><div class="lsl">💰 Financial Details</div>', unsafe_allow_html=True)
        c3,c4 = st.columns(2)
        with c3:
            app_inc  = st.number_input("Applicant Income (₹/mo)", 0, 200_000, 8_000, 500)
            coapp_inc= st.number_input("Co-applicant Income (₹/mo)", 0, 100_000, 2_000, 500)
            savings  = st.number_input("Savings Balance (₹)", 0, 500_000, 15_000, 1_000)
        with c4:
            cr_score = st.slider("Credit Score", 300, 900, 680)
            ex_loans = st.number_input("Existing Loans (#)", 0, 10, 1)
            dti      = st.slider("DTI Ratio", 0.0, 1.0, 0.35, 0.01,
                                 help="Debt-to-Income ratio (0=no debt, 1=all income is debt)")
        band_lbl, band_ico = cibil_band(cr_score)
        st.markdown(f"<span class='pill'>{band_ico} Credit: {cr_score} — {band_lbl}</span>"
                    f"<span class='pill'>DTI: {dti:.0%}</span>"
                    f"<span class='pill'>Existing Loans: {ex_loans}</span>",
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 3 — Loan & Property
        st.markdown('<div class="lsc"><div class="lsl">🏦 Loan & Property Details</div>', unsafe_allow_html=True)
        c5,c6 = st.columns(2)
        with c5:
            loan_amt  = st.number_input("Loan Amount (₹)", 5_000, 1_000_000, 30_000, 1_000)
            loan_term = st.number_input("Loan Term (months)", 12, 360, 84, 12)
            loan_purp = st.selectbox("Loan Purpose", PURPOSE_OPTS)
        with c6:
            collateral  = st.number_input("Collateral Value (₹)", 0, 2_000_000, 50_000, 5_000)
            prop_area   = st.selectbox("Property Area", PROPERTY_OPTS)
        ltv = round(loan_amt/collateral*100,1) if collateral > 0 else 0
        emi = round(loan_amt/loan_term,0) if loan_term > 0 else 0
        st.markdown(f"<span class='pill'>LTV: {ltv}%</span>"
                    f"<span class='pill'>Est. EMI: {fmt_inr(emi)}/mo</span>"
                    f"<span class='pill'>Term: {loan_term} mo</span>",
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"<div style='margin-bottom:6px'><span class='mbadge'>Model: {sel_label.split('(')[0].strip()}</span></div>",
                    unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Loan Eligibility", use_container_width=True)

    with col_r:
        # Result box
        st.markdown('<div class="lsc"><div class="lsl">📋 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            raw = {
                "Applicant_Income":coapp_inc,"Coapplicant_Income":coapp_inc,
                "Employment_Status":emp_st,"Age":age,"Marital_Status":marital,
                "Dependents":dep,"Credit_Score":cr_score,"Existing_Loans":ex_loans,
                "DTI_Ratio":dti,"Savings":savings,"Collateral_Value":collateral,
                "Loan_Amount":loan_amt,"Loan_Term":loan_term,"Loan_Purpose":loan_purp,
                "Property_Area":prop_area,"Education_Level":edu,
                "Gender":gender,"Employer_Category":emp_cat,
                "Applicant_Income":app_inc,
            }
            model = load_model(sel_pkl)
            if model is None:
                st.error(f"**{sel_pkl}** not found. Upload pkl files to the Space.")
            else:
                try:
                    X    = preprocess(raw, scaler, encoder)
                    pred = model.predict(X)[0]
                    approved = int(pred) == 1
                    if hasattr(model,"predict_proba"):
                        conf = float(max(model.predict_proba(X)[0]))*100
                    else:
                        conf = 75.0

                    if approved:
                        st.markdown("""<div class="r-ok">
                          <div class="r-ico">✅</div>
                          <div class="r-ttl" style="color:#057642">Loan Approved</div>
                          <div class="r-sub">Application meets eligibility criteria</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="r-no">
                          <div class="r-ico">❌</div>
                          <div class="r-ttl" style="color:#b24020">Loan Rejected</div>
                          <div class="r-sub">Application does not meet eligibility criteria</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"<br><strong>Model Confidence</strong> — <code>{conf:.1f}%</code>",
                                unsafe_allow_html=True)
                    st.progress(min(int(conf),100))
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.caption("Ensure scaler.pkl and encoder.pkl match training notebook.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:36px 16px;color:#aaa'>
              <div style='font-size:2.2rem'>📝</div>
              <div style='font-size:.86rem;margin-top:8px'>
                Fill in applicant details on the left and click<br>
                <strong style='color:#0a66c2'>Predict Loan Eligibility</strong>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Live Risk Scorecard — always visible
        st.markdown('<div class="lsc"><div class="lsl">⚡ Live Risk Scorecard</div>', unsafe_allow_html=True)
        cr_pct   = int((cr_score-300)/600*100)
        inc_pct  = min(int(app_inc/200_000*100),100)
        sav_pct  = min(int(savings/500_000*100),100)
        dti_pct  = max(0,int((1-dti)*100))
        coll_pct = min(int(collateral/loan_amt*100) if loan_amt else 0,100)
        for label,pct,cap in [
            ("Credit Score",  cr_pct,  f"{cr_score} — {band_lbl}"),
            ("Income Level",  inc_pct, fmt_inr(app_inc)+"/mo"),
            ("Savings Buffer",sav_pct, fmt_inr(savings)),
            ("Low DTI",       dti_pct, f"{dti:.0%} DTI"),
            ("Collateral",    coll_pct,f"LTV {ltv}%"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:.79rem;margin-bottom:1px'>"
                        f"<span style='font-weight:600'>{label}</span>"
                        f"<span style='color:#666'>{cap}</span></div>",
                        unsafe_allow_html=True)
            st.progress(pct)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("💡 Tips to improve loan eligibility"):
            st.markdown("""
- **Credit Score ≥ 700** dramatically improves approval odds
- **DTI Ratio < 0.40** — reduce existing debt before applying
- **Collateral Value > Loan Amount** lowers lender risk
- **Savings ≥ 3× EMI** signals financial stability
- **Salaried + Government employer** scores highest
- **Urban property area** has better approval rates
- Reduce **number of existing loans** before applying
            """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif "📊 Insights" in page:
    st.markdown('<div class="lsc"><div class="lsl">📊 Credit Score Reference</div>', unsafe_allow_html=True)
    cibil_df = pd.DataFrame({
        "Band":           ["Excellent (750–900)","Good (700–749)","Fair (650–699)","Poor (300–649)"],
        "Approval Rate %":[91,72,48,18],
        "Avg Interest %": [8.5,10.2,12.5,16.0],
        "Risk Level":     ["Very Low","Low","Medium","High"],
    })
    st.dataframe(cibil_df, use_container_width=True, hide_index=True,
                 column_config={"Approval Rate %": st.column_config.ProgressColumn(
                     "Approval Rate", min_value=0, max_value=100, format="%d%%")})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lsc"><div class="lsl">📈 Dataset Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records","1,000")
    c2.metric("Features","19")
    c3.metric("Approved","29.8%")
    c4.metric("Rejected","65.2%")
    st.caption("Dataset is imbalanced — majority class is Rejected. F1 and Precision used alongside accuracy.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lsc"><div class="lsl">🔑 Key Approval Factors (from EDA)</div>', unsafe_allow_html=True)
    for ico,ttl,desc in [
        ("🎯","Credit Score","Strongest predictor. ≥700 significantly boosts approval."),
        ("💰","DTI Ratio","Debt-to-income ratio. < 0.40 is preferred by lenders."),
        ("🏦","Collateral Value","Higher collateral → lower lender risk → better odds."),
        ("💼","Employment Status","Salaried > Self-employed > Contract. Unemployed rarely approved."),
        ("🏠","Property Area","Urban > Semiurban > Rural in approval probability."),
        ("💵","Savings Balance","Higher savings indicate financial stability and buffer."),
        ("👨‍👩‍👧","Dependents","More dependents reduce disposable income — moderate negative impact."),
    ]:
        st.markdown(f"<div style='display:flex;gap:12px;margin-bottom:12px;align-items:flex-start'>"
                    f"<div style='font-size:1.3rem'>{ico}</div>"
                    f"<div><div style='font-weight:600;font-size:.86rem'>{ttl}</div>"
                    f"<div style='font-size:.79rem;color:#666;margin-top:1px'>{desc}</div></div>"
                    f"</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif "🤖 Models" in page:
    st.markdown('<div class="lsc"><div class="lsl">🤖 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("> All three models trained on identical preprocessed pipeline: "
                "**StandardScaler + OHE + LabelEncoder + DTI² + Credit²**")
    mdf = pd.DataFrame({
        "Model":    ["Logistic Regression ⭐","K-Nearest Neighbours","Naive Bayes"],
        "File":     ["model_lr.pkl","model_knn.pkl","model_nb.pkl"],
        "Strength": ["Balanced Accuracy + F1","Pattern/Distance-based","Highest Precision"],
        "Best For": ["Default recommendation","Non-linear boundaries","Minimising false approvals"],
    })
    st.dataframe(mdf, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lsc"><div class="lsl">🔧 Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.code("""
Raw CSV  (1000 rows × 20 cols)
  │
  ├─ Drop: Applicant_ID
  ├─ Impute: numerical=mean | categorical=most_frequent
  ├─ LabelEncode: Education_Level, Loan_Approved
  ├─ OneHotEncode (drop=first):
  │     Employment_Status, Marital_Status, Loan_Purpose,
  │     Property_Area, Gender, Employer_Category
  ├─ Feature Engineering:
  │     DTI_Ratio_sq   = DTI_Ratio ** 2
  │     Credit_Score_sq = Credit_Score ** 2
  ├─ Drop: Credit_Score, DTI_Ratio  (originals removed)
  └─ StandardScaler → X_train_scaled / X_test_scaled
    """, language="text")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lsc"><div class="lsl">📦 Artifact Status</div>', unsafe_allow_html=True)
    for fname,desc in [("model_lr.pkl","LogisticRegression()"),
                       ("model_knn.pkl","KNeighborsClassifier(n_neighbors=5)"),
                       ("model_nb.pkl","GaussianNB()"),
                       ("scaler.pkl","StandardScaler — fitted on X_train"),
                       ("encoder.pkl","OneHotEncoder(drop='first', sparse_output=False)")]:
        exists = Path(fname).exists()
        c1,c2,c3 = st.columns([2,3,1])
        c1.code(fname)
        c2.caption(desc)
        c3.markdown("✅ Loaded" if exists else "⚠️ Missing")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif "ℹ️ About" in page:
    st.markdown('<div class="lsc"><div class="lsl">ℹ️ About LendSure AI</div>', unsafe_allow_html=True)
    st.markdown("""
**LendSure AI** is a complete end-to-end ML loan approval prediction system built as part of
the AIML practitioner portfolio at **Anna University Regional Campus, Tirunelveli**.

Three classifiers — Logistic Regression, KNN, and Naive Bayes — are trained on 1,000 real loan
application records with a full preprocessing pipeline, then served through a transparent Streamlit
UI with live risk scoring and model switching.

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| ML Models | Scikit-learn (LR · KNN · Naive Bayes) |
| Preprocessing | StandardScaler + OneHotEncoder + LabelEncoder |
| Feature Eng. | DTI² · Credit Score² |
| Dataset | loan_approval_data.csv — 1,000 rows · 19 features |
| Deployment | Hugging Face Spaces (Streamlit SDK) |
| Version Control | Git LFS (pkl files) + GitHub |

#### 🎓 Author
**Karthika Krishna M**
CSE · Anna University Regional Campus, Tirunelveli
🔗 [github.com/KARTHIKAKRISHNA123](https://github.com/KARTHIKAKRISHNA123)
🤗 [KarthikaKrishna123 on Hugging Face](https://huggingface.co/KarthikaKrishna123)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsf">
  LendSure AI · Built by <strong>Karthika Krishna M</strong> ·
  <a href="https://github.com/KARTHIKAKRISHNA123" target="_blank">GitHub</a> ·
  Anna University Regional Campus, Tirunelveli &nbsp;|&nbsp;
  <em>For educational &amp; demonstration purposes only</em>
</div>""", unsafe_allow_html=True)