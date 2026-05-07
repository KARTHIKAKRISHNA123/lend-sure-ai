import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. AGGRESSIVE THEME OVERRIDE (LinkedIn & PayPal Blue) ---
st.set_page_config(page_title="LendSure AI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    /* Force Blue Header & Background */
    [data-testid="stHeader"] { background-color: #003087 !important; color: white !important; }
    .stApp { background-color: #F3F6F9 !important; }
    
    /* PayPal Blue Headlines */
    h1 { color: #003087 !important; font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 800 !important; }
    h3 { color: #0077B5 !important; font-weight: 600 !important; }
    
    /* LinkedIn Blue Button */
    .stButton>button {
        background-color: #0077B5 !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
        font-weight: bold !important;
        height: 3.5em !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    .stButton>button:hover { background-color: #003087 !important; transform: translateY(-2px); }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 2px solid #0077B5 !important; }
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
col_logo, col_text = st.columns([1, 5])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
with col_text:
    st.title("LendSure AI")
    st.write("SecureTrust Bank: Intelligent Credit Risk Assessment")

st.divider()

# Sidebar Leaderboard
st.sidebar.markdown("<h2 style='color: #003087;'>🏆 Performance</h2>", unsafe_allow_html=True)
st.sidebar.markdown("1. **Naive Bayes** (86% Acc) ⭐\n2. **Log Reg** (84% Acc)\n3. **KNN** (81% Acc)")
selected_model = st.sidebar.selectbox("Select Model", ["Naive Bayes", "Logistic Regression", "KNN"])

# --- 4. INPUTS (9 Raw Fields) ---
st.write("### 📝 Enter Applicant Data")
c1, c2, c3 = st.columns(3)
with c1:
    income = st.number_input("Monthly Income (₹)", value=60000)
    co_income = st.number_input("Co-applicant Income (₹)", value=20000)
    loan_amt = st.number_input("Loan Amount (₹)", value=120000)
with c2:
    score = st.slider("Credit Score", 300, 900, 800)
    dti = st.slider("DTI Ratio", 0.0, 1.0, 0.20)
    age = st.number_input("Age", 18, 90, 35)
with c3:
    emp = st.selectbox("Employment", ['Salaried', 'Self-employed', 'Unemployed', 'Contract', 'Business'])
    edu = st.selectbox("Education", ['Graduate', 'Undergraduate', 'Not Graduate'])
    prop = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# --- 5. DATA PIPELINE (Building the 28 Features) ---
if st.button("RUN ASSESSMENT"):
    # Step A: Numerical Mapping for Education
    edu_numeric = {"Graduate": 2, "Undergraduate": 1, "Not Graduate": 0}[edu]

    # Step B: One-Hot Encoding
    # Recreating the exact categories used during Fit
    cat_df = pd.DataFrame([[emp, 'Married', 'Home', prop, 'Male', 'Private']], 
                          columns=['Employment_Status', 'Marital_Status', 'Loan_Purpose', 
                                   'Property_Area', 'Gender', 'Employer_Category'])
    cat_encoded = one.transform(cat_df)
    cat_final_df = pd.DataFrame(cat_encoded, columns=one.get_feature_names_out())

    # Step C: Engineered Features
    dti_sq = dti ** 2
    score_sq = score ** 2
    income_log = np.log(income) if income > 0 else 0

    # Step D: Construct the 28-column Vector
    # Numeric Inputs (10 columns)
    numeric_data = [income, co_income, age, 1, 0, 25000, 100000, loan_amt, 72, edu_numeric]
    numeric_cols = ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 
                    'Existing_Loans', 'Savings', 'Collateral_Value', 'Loan_Amount', 
                    'Loan_Term', 'Education_Level']
    
    final_df = pd.DataFrame([numeric_data], columns=numeric_cols)
    
    # Concatenate Numeric + Encoded (25 columns total)
    final_df = pd.concat([final_df, cat_final_df], axis=1)
    
    # Add Engineered (28 columns total)
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

    # Results UI
    st.markdown("---")
    if prediction == 1:
        st.success("## ✅ RESULT: LOAN APPROVED")
        st.balloons()
        st.write("This applicant meets SecureTrust Bank's safety profile.")
    else:
        st.error("## ❌ RESULT: LOAN REJECTED")
        st.write("High risk detected. Consider improving credit score or lowering DTI ratio.")