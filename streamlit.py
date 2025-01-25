import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load the trained model
model_file = "credit_score_xgb_model.pkl"
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Streamlit app header
st.title("Credit Score Prediction")

# Instructions for the user
st.write("Please enter the details below to predict the credit score.")

# Input form for the user
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
daily_wage = st.number_input("Daily Wage", min_value=0, value=200)
household_size = st.number_input("Household Size", min_value=1, value=5)
land_ownership = st.selectbox("Land Ownership", options=[0, 1], index=1)
loan_repayment_history = st.selectbox("Loan Repayment History", options=[0, 1], index=0)
savings = st.number_input("Savings", min_value=0, value=1000)
number_of_loans = st.number_input("Number of Loans", min_value=0, value=2)
loan_amount = st.number_input("Loan Amount", min_value=0, value=15000)
community_reliability_rating = st.number_input("Community Reliability Rating", min_value=1, value=4)
participation_in_MGNREGA = st.selectbox("Participation in MGNREGA", options=[0, 1], index=1)
mobile_phone_cost = st.number_input("Mobile Phone Cost", min_value=0, value=50)
health_expenditure = st.number_input("Health Expenditure", min_value=0, value=200)
child_education_expenditure = st.number_input("Child Education Expenditure", min_value=0, value=150)
crop_yield = st.number_input("Crop Yield", min_value=0, value=1200)
vehicle_owned = st.selectbox("Vehicle Owned", options=[0, 1], index=1)
utility_bill_payment = st.number_input("Utility Bill Payment", min_value=0, value=300)
insurance_coverage = st.selectbox("Insurance Coverage", options=[0, 1], index=1)
daily_spent_food = st.number_input("Daily Spent on Food", min_value=0, value=50)
daily_spent_travel = st.number_input("Daily Spent on Travel", min_value=0, value=20)
daily_spent_health = st.number_input("Daily Spent on Health", min_value=0, value=30)
daily_spent_savings = st.number_input("Daily Spent on Savings", min_value=0, value=10)

# Create a DataFrame from user inputs
user_input = {
    "annual_income": annual_income,
    "daily_wage": daily_wage,
    "household_size": household_size,
    "land_ownership": land_ownership,
    "loan_repayment_history": loan_repayment_history,
    "savings": savings,
    "number_of_loans": number_of_loans,
    "loan_amount": loan_amount,
    "community_reliability_rating": community_reliability_rating,
    "participation_in_MGNREGA": participation_in_MGNREGA,
    "mobile_phone_cost": mobile_phone_cost,
    "health_expenditure": health_expenditure,
    "child_education_expenditure": child_education_expenditure,
    "crop_yield": crop_yield,
    "vehicle_owned": vehicle_owned,
    "utility_bill_payment": utility_bill_payment,
    "insurance_coverage": insurance_coverage,
    "daily_spent_food": daily_spent_food,
    "daily_spent_travel": daily_spent_travel,
    "daily_spent_health": daily_spent_health,
    "daily_spent_savings": daily_spent_savings,
}

input_df = pd.DataFrame([user_input])

# Handle categorical variables (one-hot encoding)
input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure the input features match the model's training features
model_features = model.feature_names  # Features used during model training
missing_columns = set(model_features) - set(input_df.columns)

for col in missing_columns:
    input_df[col] = 0  # Add missing columns with default value 0

input_df = input_df[model_features]  # Reorder the columns to match the model

# Convert the input DataFrame to DMatrix for prediction
dinput = xgb.DMatrix(input_df)

# Predict the credit score using the trained model
if st.button("Predict Credit Score"):
    predicted_credit_score = model.predict(dinput)
    st.write(f"**Predicted Credit Score:** {predicted_credit_score[0]:.2f}")
