import streamlit as st
import pandas as pd
import joblib

# Load the trained model and model columns
model = joblib.load("rf_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("💼 Term Deposit Subscription Predictor")
st.markdown("Enter client details to predict the likelihood of subscribing to a term deposit.")

# ====== USER INPUTS ======

age = st.number_input("Client Age", min_value=18, max_value=100, value=35)

campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=1)

pdays = st.number_input("Days Since Last Contact (999 = Never Contacted)", min_value=0, max_value=999, value=999)

euribor3m = st.number_input("3-Month Euribor Rate", format="%.2f", value=4.85)

emp_var_rate = st.number_input("Employment Variation Rate", format="%.2f", value=-1.8)

#nr_employed = st.number_input("Number of People Employed", format="%.1f", value=5099.1)

housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])

loan = st.selectbox("Has Personal Loan?", ['yes', 'no'])

# ====== PREPROCESSING ======

# Build input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'campaign': [campaign],
    'pdays': [pdays],
    'euribor3m': [euribor3m],
    'emp.var.rate': [emp_var_rate],
    #'nr.employed': [nr_employed],
    'housing_yes': [1 if housing == 'yes' else 0],
    'loan_yes': [1 if loan == 'yes' else 0]
})

# Fill in missing columns with 0s (required for consistency with training)
for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match training data
input_data = input_data[columns]

# ====== PREDICTION ======
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"✅ Likely to Subscribe (probability: {prob:.2%})")
    else:
        st.error(f"❌ Not Likely to Subscribe (probability: {prob:.2%})")
