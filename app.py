import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. PAGE CONFIG & INDUSTRY THEME (LinkedIn/PayPal) ---
st.set_page_config(page_title="LendSure AI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    /* LinkedIn/PayPal Style Theme */
    .main { background-color: #F3F6F9; } 
    h1 { color: #003087; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 800; text-align: center; }
    h3 { color: #0077B5; font-weight: 600; }
    
    /* Rounded PayPal-style Button */
    .stButton>button {
        background-color: #0077B5;
        color: white;
        border-radius: 50px; 
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
        height: 3.5em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #003087; transform: scale(1.02); }
    
    /* Input Container Styling */
    .css-1r6slb0 { padding: 2.5rem; border-radius: 15px; background: white; border: 1px solid #E1E9EE; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS & ASSETS ---
@st.cache_resource
def load_assets():
    # Asset names match your notebook exports exactly 
    nb = pickle.load(open('model_nb.pkl', 'rb'))
    lr = pickle.load(open('model_lr.pkl', 'rb'))
    knn = pickle.load(open('model_knn.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    # User requested to keep this variable name as 'ohe'
    ohe = pickle.load(open('encoder.pkl', 'rb')) 
    return nb, lr, knn, scaler, ohe

nb_model, log_model, knn_model, scaler, ohe = load_assets()

# --- 3. UI LOGO & HEADER ---
col_l, col_r = st.columns([1, 4])
with col_l:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=120)
with col_r:
    st.title("LendSure AI")
    st.markdown("<p style='color: #666; font-size: 1.2rem; margin-top: -20px;'>Intelligent Credit Risk Assessment System for SecureTrust Bank</p>", unsafe_allow_html=True)

st.divider()

# --- 4. SIDEBAR RANKING ---
st.sidebar.markdown("<h2 style='color: #003087;'>🏆 Leaderboard</h2>", unsafe_allow_html=True)
st.sidebar.info("""
1. **Naive Bayes** (86% Acc) ⭐
2. **Logistic Regression** (84% Acc)
3. **K-Nearest Neighbors** (81% Acc)
""")
selected_model = st.sidebar.selectbox("Choose Prediction Logic", ["Naive Bayes", "Logistic Regression", "KNN"])

# --- 5. APPLICANT INPUTS ---
st.write("### 📋 Applicant Financial Profile")
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        income = st.number_input("Monthly Income (₹)", value=35000)
        co_income = st.number_input("Co-applicant Income (₹)", value=15000)
        loan_amt = st.number_input("Requested Loan Amount (₹)", value=150000)
        loan_term = st.number_input("Loan Term (Months)", value=60)
    with c2:
        score = st.slider("CIBIL/Credit Score", 300, 900, 750)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.35)
        age = st.number_input("Applicant Age", 18, 90, 28)
        dependents = st.number_input("Number of Dependents", 0, 10, 0)
    with c3:
        existing_loans = st.number_input("Existing Loans", 0, 10, 1)
        savings = st.number_input("Savings Balance (₹)", value=10000)
        collateral = st.number_input("Collateral Value (₹)", value=50000)
        
st.write("### 🏠 Lifestyle & Background")
col_cat1, col_cat2, col_cat3 = st.columns(3)
with col_cat1:
    emp = st.selectbox("Employment", ['Salaried', 'Self-employed', 'Unemployed', 'Contract', 'Business'])
    edu = st.selectbox("Education Level", ['Graduate', 'Undergraduate', 'Not Graduate'])
with col_cat2:
    prop = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
with col_cat3:
    marital = st.selectbox("Marital Status", ['Married', 'Single'])
    purpose = st.selectbox("Loan Purpose", ['Home', 'Education', 'Personal', 'Business', 'Car'])
    emp_cat = st.selectbox("Employer Category", ['Private', 'Government', 'MNC', 'Unemployed'])

# --- 6. PREDICTION LOGIC ---
if st.button("RUN ASSESSMENT"):
    # Step A: Numerical Mapping for Education (Matches Notebook Cell 15) 
    edu_map = {"Graduate": 2, "Undergraduate": 1, "Not Graduate": 0}
    edu_numeric = edu_map[edu]

    # Step B: One-Hot Encoding for Categorical columns (Matches encoder.pkl) [cite: 9]
    # 'Education_Level' was NOT part of your OneHotEncoder [cite: 9]
    cat_cols_fit = ['Employment_Status', 'Marital_Status', 'Loan_Purpose', 'Property_Area', 'Gender', 'Employer_Category']
    cat_data = pd.DataFrame([[emp, marital, purpose, prop, gender, emp_cat]], columns=cat_cols_fit)
    cat_encoded = ohe.transform(cat_data)
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols_fit))

    # Step C: Feature Engineering (Matches Notebook Cell 18) 
    dti_sq = dti ** 2
    score_sq = score ** 2
    # Ensure income is positive for log
    income_log = np.log(income) if income > 0 else 0

    # Step D: Construct the Final 28-Feature Vector in the Correct Order 
    # 1-10: Numeric
    final_input = pd.DataFrame([[
        income, co_income, age, dependents, existing_loans, 
        savings, collateral, loan_amt, loan_term, edu_numeric
    ]], columns=[
        'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 
        'Existing_Loans', 'Savings', 'Collateral_Value', 'Loan_Amount', 
        'Loan_Term', 'Education_Level'
    ])

    # 11-25: OHE Categorical
    final_input = pd.concat([final_input, cat_df], axis=1)

    # 26-28: Engineered Features
    final_input['DTI_Ratio_sq'] = dti_sq
    final_input['Credit_Score_sq'] = score_sq
    final_input['Applicant_Income_log'] = income_log

    # Step E: Scaling & Prediction 
    final_input_scaled = scaler.transform(final_input)

    with st.spinner('LendSure AI is assessing risk...'):
        if selected_model == "Naive Bayes":
            pred = nb_model.predict(final_input_scaled)[0]
        elif selected_model == "Logistic Regression":
            pred = log_model.predict(final_input_scaled)[0]
        else:
            pred = knn_model.predict(final_input_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.success("## ✅ RESULT: LOAN APPROVED")
            st.balloons()
            st.write("Applicant meets the safety thresholds for SecureTrust Bank.")
        else:
            st.error("## ❌ RESULT: LOAN REJECTED")
            st.write("Analysis indicates a high-risk profile for default.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for SecureTrust Bank AI Operations ")