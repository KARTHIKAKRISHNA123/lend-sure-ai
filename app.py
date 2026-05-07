import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. THEME: LinkedIn (#0077B5) & PayPal (#003087) ---
st.set_page_config(page_title="LendSure AI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    h1 { color: #003087 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    h3 { color: #0077B5 !important; }
    .stButton>button {
        background-color: #0077B5 !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        font-weight: bold !important;
        width: 100% !important;
        height: 3em !important;
    }
    .stButton>button:hover { background-color: #003087 !important; }
    /* Styling for the sidebar leaderboard */
    [data-testid="stSidebar"] { background-color: #F3F6F9 !important; border-right: 1px solid #E1E9EE; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Variable names match your notebook exactly
    nb = pickle.load(open('model_nb.pkl', 'rb'))
    lr = pickle.load(open('model_lr.pkl', 'rb'))
    knn = pickle.load(open('model_knn.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    one = pickle.load(open('encoder.pkl', 'rb')) # Variable 'one' from notebook
    return nb, lr, knn, scaler, one

nb_model, log_model, knn_model, scaler, one = load_assets()

# --- 3. UI LAYOUT ---
st.title("🏦 LendSure AI")
st.markdown("<p style='color: #666;'>SecureTrust Bank: Intelligent Credit Risk Assessment</p>", unsafe_allow_html=True)
st.divider()

# Sidebar Leaderboard
st.sidebar.markdown("<h2 style='color: #003087;'>🏆 Leaderboard</h2>", unsafe_allow_html=True)
st.sidebar.info("🥇 **Naive Bayes**: 86.0% Acc\n🥈 **Log Reg**: 84.0% Acc\n🥉 **KNN**: 81.7% Acc")
selected_model = st.sidebar.selectbox("Choose AI Engine", ["Naive Bayes", "Logistic Regression", "KNN"])

# --- 4. INPUTS (9 Raw Fields) ---
st.write("### 📋 Applicant Details")
c1, c2, c3 = st.columns(3)
with c1:
    income = st.number_input("Monthly Income (₹)", value=45000)
    co_income = st.number_input("Co-applicant Income (₹)", value=20000)
    loan_amt = st.number_input("Loan Amount (₹)", value=100000)
with c2:
    score = st.slider("Credit Score", 300, 900, 750)
    dti = st.slider("DTI Ratio", 0.0, 1.0, 0.25)
    age = st.number_input("Age", 18, 90, 32)
with c3:
    emp = st.selectbox("Employment", ['Salaried', 'Self-employed', 'Unemployed', 'Contract', 'Business'])
    edu = st.selectbox("Education", ['Graduate', 'Undergraduate', 'Not Graduate'])
    prop = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# --- 5. DATA PIPELINE (Building the 28 Features) ---
if st.button("RUN ASSESSMENT"):
    # Step A: Numerical Mapping for Education
    edu_numeric = {"Graduate": 2, "Undergraduate": 1, "Not Graduate": 0}[edu]

    # Step B: One-Hot Encoding for remaining categories
    # NOTE: scaler.pkl shows Gender, Marital_Status, etc. were part of training
    cat_df = pd.DataFrame([[emp, 'Married', 'Home', prop, 'Male', 'Private']], 
                          columns=['Employment_Status', 'Marital_Status', 'Loan_Purpose', 
                                   'Property_Area', 'Gender', 'Employer_Category'])
    cat_encoded = one.transform(cat_df)
    cat_cols_encoded = one.get_feature_names_out()
    cat_final_df = pd.DataFrame(cat_encoded, columns=cat_cols_encoded)

    # Step C: Engineered Features
    dti_sq = dti ** 2
    score_sq = score ** 2
    income_log = np.log(income) if income > 0 else 0

    # Step D: Construct the 28-column Vector
    # Must follow the EXACT order in scaler.pkl: Numeric, then Encoded, then Engineered
    numeric_data = [income, co_income, age, 1, 0, 15000, 50000, loan_amt, 60, edu_numeric]
    numeric_cols = ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 
                    'Existing_Loans', 'Savings', 'Collateral_Value', 'Loan_Amount', 
                    'Loan_Term', 'Education_Level']
    
    final_df = pd.DataFrame([numeric_data], columns=numeric_cols)
    final_df = pd.concat([final_df, cat_final_df], axis=1)
    
    # Add final engineered features
    final_df['DTI_Ratio_sq'] = dti_sq
    final_df['Credit_Score_sq'] = score_sq
    final_df['Applicant_Income_log'] = income_log

    # Step E: Scaling & Prediction
    final_scaled = scaler.transform(final_df)
    
    if selected_model == "Naive Bayes":
        prediction = nb_model.predict(final_scaled)[0]
    elif selected_model == "Logistic Regression":
        prediction = log_model.predict(final_scaled)[0]
    else:
        prediction = knn_model.predict(final_scaled)[0]

    # Display Results
    st.markdown("---")
    if prediction == 1:
        st.success("## ✅ RESULT: LOAN APPROVED")
        st.balloons()
    else:
        st.error("## ❌ RESULT: LOAN REJECTED")