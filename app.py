import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. DYNAMIC BLUE ANIMATED BACKGROUND & THEME ---
st.set_page_config(page_title="LendSure AI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #F3F6F9, #DCE6F1, #0077B5, #003087);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* PayPal/LinkedIn Styling */
    h1 { color: #FFFFFF !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    h3 { color: #003087 !important; background: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 10px; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.9) !important; border-right: 3px solid #0077B5; }

    /* Button Styling */
    .stButton>button {
        background-color: #0077B5 !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
        font-weight: bold !important;
        height: 3.5em !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    .stButton>button:hover { background-color: #003087 !important; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Variable names match your notebook and pickle files
    nb = pickle.load(open('model_nb.pkl', 'rb'))
    lr = pickle.load(open('model_lr.pkl', 'rb'))
    knn = pickle.load(open('model_knn.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    one = pickle.load(open('encoder.pkl', 'rb')) # Using variable 'one'
    return nb, lr, knn, scaler, one

nb_model, log_model, knn_model, scaler, one = load_assets()

# --- 3. UI LAYOUT ---
st.title("🏦 LendSure AI")
st.write("### SecureTrust Bank: Intelligent Credit Risk Assessment")
st.divider()

# Sidebar Leaderboard
st.sidebar.markdown("<h2 style='color: #003087;'>🏆 Leaderboard</h2>", unsafe_allow_html=True)
st.sidebar.info("🥇 **Naive Bayes**: 86.0% Acc\n🥈 **Log Reg**: 84.0% Acc\n🥉 **KNN**: 81.7% Acc")
selected_model = st.sidebar.selectbox("Choose AI Engine", ["Naive Bayes", "Logistic Regression", "KNN"])

# --- 4. INPUTS ---
st.write("### 📋 Applicant Financial Profile")
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        income = st.number_input("Monthly Income (₹)", value=55000)
        co_income = st.number_input("Co-applicant Income (₹)", value=25000)
        loan_amt = st.number_input("Loan Amount (₹)", value=120000)
    with c2:
        score = st.slider("Credit Score", 300, 900, 780)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.22)
        age = st.number_input("Age", 18, 90, 34)
    with c3:
        emp = st.selectbox("Employment", ['Salaried', 'Self-employed', 'Unemployed', 'Contract', 'Business'])
        edu = st.selectbox("Education", ['Graduate', 'Undergraduate', 'Not Graduate'])
        prop = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# --- 5. THE 28-FEATURE PIPELINE ---
if st.button("RUN ASSESSMENT"):
    # A. Map Education to Numeric
    edu_numeric = {"Graduate": 2, "Undergraduate": 1, "Not Graduate": 0}[edu]

    # B. One-Hot Encode (6 categories)
    # Recreating categorical order found in encoder metadata
    cat_df = pd.DataFrame([[emp, 'Married', 'Home', prop, 'Male', 'Private']], 
                          columns=['Employment_Status', 'Marital_Status', 'Loan_Purpose', 
                                   'Property_Area', 'Gender', 'Employer_Category'])
    cat_encoded = one.transform(cat_df)
    cat_final_df = pd.DataFrame(cat_encoded, columns=one.get_feature_names_out())

    # C. Calculate Engineered Features
    dti_sq = dti ** 2
    score_sq = score ** 2
    income_log = np.log(income) if income > 0 else 0

    # D. Build Final DataFrame (28 Columns in exact order)
    # Columns 1-10: Numeric
    numeric_data = [income, co_income, age, 1, 0, 30000, 80000, loan_amt, 84, edu_numeric]
    numeric_cols = ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 
                    'Existing_Loans', 'Savings', 'Collateral_Value', 'Loan_Amount', 
                    'Loan_Term', 'Education_Level']
    final_df = pd.DataFrame([numeric_data], columns=numeric_cols)
    
    # Columns 11-25: Encoded Categories
    final_df = pd.concat([final_df, cat_final_df], axis=1)
    
    # Columns 26-28: Engineered Features
    final_df['DTI_Ratio_sq'] = dti_sq
    final_df['Credit_Score_sq'] = score_sq
    final_df['Applicant_Income_log'] = income_log

    # E. Scaling & Prediction
    final_scaled = scaler.transform(final_df)
    
    # Model Selection
    model = {"Naive Bayes": nb_model, "Logistic Regression": log_model, "KNN": knn_model}[selected_model]
    prediction = model.predict(final_scaled)[0]

    # RESULTS
    st.markdown("---")
    if prediction == 1:
        st.success("## ✅ RESULT: LOAN APPROVED")
        st.balloons()
        st.write("Applicant profile meets the secure thresholds for SecureTrust Bank.")
    else:
        st.error("## ❌ RESULT: LOAN REJECTED")
        st.write("Potential risk detected. Review credit score and DTI alignment.")