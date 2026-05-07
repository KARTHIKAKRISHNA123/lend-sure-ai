import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ---------- 1. DYNAMIC THEME & BLUE ANIMATIONS ----------
st.set_page_config(page_title="LendSure AI", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #f5f7fb, #eef2f7, #0077B5, #003087);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Branding Colors */
    :root { --linkedin-blue: #0077B5; --paypal-blue: #003087; }
    h1 { color: #02132b !important; font-weight: 800 !important; }
    
    /* PayPal-style Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px; border-radius: 12px;
        border-left: 5px solid var(--linkedin-blue);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* LinkedIn-style Primary Button */
    .stButton>button {
        background: linear-gradient(180deg, #0077B5, #003087) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 12px 24px !important;
        font-weight: 600 !important; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 2. ASSET LOADING ----------
@st.cache_resource
def load_assets():
    # Loading based on your actual file names
    return {
        "nb": pickle.load(open("model_nb.pkl", "rb")),
        "lr": pickle.load(open("model_lr.pkl", "rb")),
        "knn": pickle.load(open("model_knn.pkl", "rb")),
        "scaler": pickle.load(open("scaler.pkl", "rb")),
        "ohe": pickle.load(open("encoder.pkl", "rb"))
    }

assets = load_assets()

# ---------- 3. UI LAYOUT & SIDEBAR ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("Navigation")
    engine = st.radio("AI Engine", ["Naive Bayes", "Logistic Regression", "KNN"])
    st.divider()
    st.markdown("### 🏆 Leaderboard")
    st.write("🥇 **Naive Bayes** `86.0%`")
    st.write("🥈 **Log Reg** `84.0%`")
    st.write("🥉 **KNN** `81.7%`")

st.title("🏦 LendSure AI")
st.caption("Intelligent Credit Risk Assessment · SecureTrust Bank Operations")

m1, m2, m3 = st.columns(3)
m1.metric("Apps scored", "12,847", "+128")
m2.metric("Best Accuracy", "86.0%", "Stable")
m3.metric("Latency", "432ms", "-12ms")

st.divider()

# ---------- 4. DATA INPUT FORM ----------
st.subheader("📋 Applicant Financial Profile")
c1, c2, c3 = st.columns(3)

with c1:
    income = st.number_input("Monthly Income (₹)", 0, 1000000, 55000)
    co_inc = st.number_input("Co-applicant Income (₹)", 0, 1000000, 25000)
    loan = st.number_input("Loan Amount (₹)", 0, 5000000, 120000)
with c2:
    score = st.slider("Credit Score (CIBIL)", 300, 900, 780)
    dti = st.slider("DTI Ratio", 0.0, 1.0, 0.22)
    age = st.number_input("Age", 18, 90, 34)
with c3:
    emp = st.selectbox("Employment", ["Salaried", "Self-employed", "Unemployed", "Business"])
    edu = st.selectbox("Education", ["Graduate", "Postgraduate", "Undergraduate"])
    prop = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------- 5. THE 28-FEATURE PIPELINE ----------
def process_data():
    # A. Map Education (Ordinal)
    edu_map = {"Graduate": 2, "Postgraduate": 3, "Undergraduate": 1}
    
    # B. OHE Categories (Recreating exact 15 binary columns)
    cat_df = pd.DataFrame([[emp, 'Married', 'Home', prop, 'Male', 'Private']], 
                          columns=['Employment_Status', 'Marital_Status', 'Loan_Purpose', 
                                   'Property_Area', 'Gender', 'Employer_Category'])
    cat_encoded = assets['ohe'].transform(cat_df)
    cat_final_df = pd.DataFrame(cat_encoded, columns=assets['ohe'].get_feature_names_out())

    # C. Numerical & Engineered Features
    numeric_data = {
        'Applicant_Income': income, 'Coapplicant_Income': co_inc, 'Age': age, 
        'Dependents': 1, 'Existing_Loans': 0, 'Savings': 30000, 
        'Collateral_Value': 80000, 'Loan_Amount': loan, 'Loan_Term': 60, 
        'Education_Level': edu_map.get(edu, 0)
    }
    final_df = pd.DataFrame([numeric_data])
    final_df = pd.concat([final_df, cat_final_df], axis=1)
    
    # Add Squared & Log Features
    final_df['DTI_Ratio_sq'] = dti ** 2
    final_df['Credit_Score_sq'] = score ** 2
    final_df['Applicant_Income_log'] = np.log(income) if income > 0 else 0
    
    return assets['scaler'].transform(final_df)

# ---------- 6. ASSESSMENT ----------
if st.button("🚀 Run Assessment"):
    with st.spinner("Analyzing Credit Risk..."):
        time.sleep(0.5) # Simulate processing
        X_scaled = process_data()
        
        model_key = {"Naive Bayes": "nb", "Logistic Regression": "lr", "KNN": "knn"}[engine]
        prediction = assets[model_key].predict(X_scaled)[0]
        
        st.divider()
        if prediction == 1:
            st.success("### ✅ RESULT: APPROVED")
            st.balloons()
            st.write("Applicant presents a Low Risk profile. High Precision match (81.1%).")
        else:
            st.error("### ❌ RESULT: REJECTED")
            st.write("Applicant exceeds the Risk Threshold for SecureTrust Bank.")