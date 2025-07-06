import streamlit as st
import numpy as np
import joblib


model=joblib.load('linear_regression_model.joblib')

# Collect input from user
claim_amount = st.number_input("Claim Amount", min_value=0.0, format="%.2f")
past_consultations = st.number_input("Number of Past Consultations", min_value=0)
hospital_expenditure = st.number_input("Hospital Expenditure", min_value=0.0, format="%.2f")
annual_salary = st.number_input("Annual Salary", min_value=0.0, format="%.2f")
children = st.number_input("Number of Children", min_value=0)
smoker = st.selectbox("Is the person a smoker?", ["No", "Yes"])
smoker_encoded = 1 if smoker == 'Yes' else 0

input_data = np.array([[smoker_encoded, claim_amount, past_consultations, hospital_expenditure, annual_salary, children]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Value:")
    st.write(prediction)

