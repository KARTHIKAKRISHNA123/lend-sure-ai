import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. PAGE CONFIG & THEME (LinkedIn/PayPal Style) ---
st.set_page_config(page_title="LendSure AI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F3F6F9; } /* LinkedIn light grey background */
    h1 { color: #003087; font-family: 'Arial'; font-weight: 700; } /* PayPal Blue */
    h3 { color: #0077B5; } /* LinkedIn Blue */
    .stButton>button {
        background-color: #0077B5;
        color: white;
        border-radius: 25px; /* Rounded PayPal style */
        border: none;
        font-weight: bold;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #003087; color: white; }
    .css-1r6slb0 { padding: 2rem; border-radius: 10px; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Variables match your notebook precisely
    nb = pickle.load(open('model_nb.pkl', 'rb'))
    lr = pickle.load(open('model_lr.pkl', 'rb'))
    knn = pickle.load(open('model_knn.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    ohe = pickle.load(open('encoder.pkl', 'rb'))
    return nb, lr, knn, scaler, ohe

nb, lr, knn, scaler, ohe = load_assets()

# --- 3. UI HEADER ---
st.title("🏦 LendSure AI")
st.subheader("Intelligent Credit Risk Assessment System")
st.markdown("---")

# --- 4. SIDEBAR LEADERBOARD ---
st.sidebar.markdown("<h2 style='color: white;'>🏆 Ranking</h2>", unsafe_allow_html=True)
st.sidebar.info("""
1. **Naive Bayes** (86.0% Acc) ⭐
2. **Logistic Regression** (84.2% Acc)
3. **K-Nearest Neighbors** (81.7% Acc)
""")
selected_model = st.sidebar.selectbox("Prediction Engine", ["Naive Bayes", "Logistic Regression", "KNN"])

# --- 5. INPUT FIELDS ---
st.write("### 📝 Applicant Profile")
col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Monthly Income (₹)", min_value=0, value=25000)
    co_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=10000)
    loan_amt = st.number_input("Loan Amount Requested (₹)", min_value=0, value=50000)

with col2:
    credit_score = st.slider("Credit Score", 300, 900, 700)
    dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
    age = st.number_input("Age", 18, 90, 30)

with col3:
    emp = st.selectbox("Employment", ['Salaried', 'Self-Employed', 'Business'])
    edu = st.selectbox("Education", ['Graduate', 'Postgraduate', 'Undergraduate'])
    prop = st.selectbox("Property Area", ['Urban', 'Semi-Urban', 'Rural'])

# --- 6. PREDICTION LOGIC ---
if st.button("RUN RISK ANALYSIS"):
    # Create DataFrame from inputs
    data = pd.DataFrame([[income, co_income, emp, age, credit_score, dti, loan_amt, edu, prop]], 
                        columns=['Applicant_Income', 'Coapplicant_Income', 'Employment_Status', 
                                 'Age', 'Credit_Score', 'DTI_Ratio', 'Loan_Amount', 
                                 'Education_Level', 'Property_Area'])

    # Step A: Preprocess Categorical (This expands 3 columns into many, matching training)
    cat_cols = ['Employment_Status', 'Education_Level', 'Property_Area']
    cat_encoded = ohe.transform(data[cat_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols))

    # Step B: Combine with Numeric & Scale
    num_df = data.drop(columns=cat_cols)
    final_input = pd.concat([num_df, cat_df], axis=1)
    final_input_scaled = scaler.transform(final_input) # This results in exactly 28 features!

    with st.spinner('Calculating Eligibility...'):
        if selected_model == "Naive Bayes":
            pred = nb.predict(final_input_scaled)[0]
        elif selected_model == "Logistic Regression":
            pred = lr.predict(final_input_scaled)[0]
        else:
            pred = knn.predict(final_input_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.success("## ✅ RESULT: LOAN APPROVED")
            st.balloons()
        else:
            st.error("## ❌ RESULT: LOAN REJECTED")
            st.write("**Risk Analysis:** Low credit score or high DTI ratio detected.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for SecureTrust Bank")