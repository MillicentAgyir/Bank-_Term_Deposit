import streamlit as st
import pandas as pd
import joblib

# Load trained model and column structure
model = joblib.load("rf_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("💼 Term Deposit Subscription Predictor")

st.markdown("Enter client information to predict whether they will subscribe to a term deposit.")

# Input fields for selected features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
campaign = st.number_input("Number of Campaign Contacts", min_value=1, value=1)
euribor3m = st.number_input("3-Month Euribor Rate", value=4.85)
housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
loan = st.selectbox("Has Personal Loan?", ["yes", "no"])

# Prepare input
input_data = pd.DataFrame({
    'age': [age],
    'campaign': [campaign],
    'euribor3m': [euribor3m],
    'housing_yes': [1 if housing == 'yes' else 0],
    'loan_yes': [1 if loan == 'yes' else 0]
})

# Add missing columns as 0s (to match training set)
for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[columns]

# Predict
if st.button("Predict"):
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if result == 1:
        st.success(f"✅ Likely to Subscribe (probability: {prob:.2f})")
    else:
        st.error(f"❌ Not Likely to Subscribe (probability: {prob:.2f})")
