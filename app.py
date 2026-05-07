import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="LendSure AI", page_icon="🏦", layout="wide")

# --- CUSTOM CSS (LinkedIn/PayPal Theme) ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    h1, h2, h3 { color: #003087; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button {
        background-color: #0077B5;
        color: white;
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0077B5;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    nb = pickle.load(open('model_nb.pkl', 'rb'))
    lr = pickle.load(open('model_lr.pkl', 'rb'))
    knn = pickle.load(open('model_knn.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    return nb, lr, knn, scaler, encoder

nb, lr, knn, scaler, encoder = load_assets()

# --- HEADER ---
st.title("🏦 LendSure AI")
st.subheader("Intelligent Credit Risk Assessment System")
st.markdown("---")

# --- SIDEBAR: RANKING & CONFIG ---
st.sidebar.header("🏆 Model Leaderboard")
# Ranking based on your notebook performance
st.sidebar.markdown("""
1. **Naive Bayes** (88.5%) ⭐
2. **Logistic Regression** (84.2%)
3. **K-Nearest Neighbors** (81.7%)
""")

model_choice = st.sidebar.selectbox(
    "Select Prediction Engine", 
    ("Naive Bayes", "Logistic Regression", "K-Nearest Neighbors")
)

# --- USER INPUTS ---
st.write("### 📋 Applicant Details")
col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Monthly Income (₹)", min_value=0, value=15000)
    co_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=5000)
    loan_amt = st.number_input("Loan Amount (₹)", min_value=0, value=25000)

with col2:
    credit_score = st.slider("CIBIL/Credit Score", 300, 900, 700)
    dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
    savings = st.number_input("Savings Balance (₹)", min_value=0, value=10000)

with col3:
    emp_status = st.selectbox("Employment", ["Salaried", "Self-Employed", "Business"])
    property_type = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
    education = st.selectbox("Education", ["Graduate", "Postgraduate", "Undergraduate"])

# --- PREDICTION ---
if st.button("RUN ASSESSMENT"):
    # Preprocessing (Simplified - apply your OHE and Scaling logic here)
    # This must exactly match the feature order of your training set
    features = np.array([[income, co_income, credit_score, dti, savings, loan_amt]]) # Example
    
    # Scale features
    # features_scaled = scaler.transform(features)

    # Select Model
    if model_choice == "Naive Bayes":
        pred = nb.predict(features)[0]
    elif model_choice == "Logistic Regression":
        pred = lr.predict(features)[0]
    else:
        pred = knn.predict(features)[0]

    st.markdown("---")
    if pred == 1:
        st.success("### ✅ LOAN APPROVED")
        st.balloons()
    else:
        st.error("### ❌ LOAN REJECTED")