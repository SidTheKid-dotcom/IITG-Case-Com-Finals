import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model_file = "credit_score_model.pkl"
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Define the function to predict the credit score
def predict_credit_score(input_data):
    # Convert input data to DataFrame (matching the model's input format)
    input_df = pd.DataFrame([input_data])

    # One-hot encoding for categorical variables (since the model was trained with this)
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Get the list of features the model was trained on
    model_features = model.feature_importances_

    # Ensure the columns match with the training data by adding any missing columns
    missing_columns = set(model_features) - set(input_df.columns)
    for col in missing_columns:
        input_df[col] = 0  # Add missing columns with a default value (0 for numerical)

    # Reorder columns to match the model's expected order
    input_df = input_df[model_features]

    # Predict the credit score
    predicted_score = model.predict(input_df)[0]
    return predicted_score

# Streamlit app layout
st.title("Credit Score Prediction App")

st.write("This app predicts the credit score based on the input data provided.")

# Mobile-centric view (Streamlit handles responsive design by default)
input_data = {
    "annual_income": st.number_input("Annual Income (₹)", min_value=0, value=500000),
    "daily_wage": st.number_input("Daily Wage (₹)", min_value=0, value=300),
    "household_size": st.number_input("Household Size", min_value=1, value=5),
    "land_ownership": st.selectbox("Land Ownership", options=["Rented", "Owned"], index=1),
    "loan_repayment_history": st.number_input("Loan Repayment History (in months)", min_value=0, value=8),
    "savings": st.number_input("Savings (₹)", min_value=0, value=20000),
    "number_of_loans": st.number_input("Number of Loans", min_value=0, value=2),
    "loan_amount": st.number_input("Loan Amount (₹)", min_value=0, value=100000),
    "community_reliability_rating": st.number_input("Community Reliability Rating", min_value=1, max_value=5, value=4),
    "participation_in_MGNREGA": st.number_input("Participation in MGNREGA (%)", min_value=0, max_value=100, value=75),
    "mobile_phone_cost": st.number_input("Mobile Phone Cost (₹)", min_value=0, value=500),
    "health_expenditure": st.number_input("Health Expenditure (₹)", min_value=0, value=1500),
    "child_education_expenditure": st.number_input("Child Education Expenditure (₹)", min_value=0, value=2000),
    "crop_yield": st.number_input("Crop Yield (₹)", min_value=0, value=1500),
    "vehicle_owned": st.selectbox("Vehicle Owned", options=["No", "Yes"], index=1),
    "utility_bill_payment": st.number_input("Utility Bill Payment (₹)", min_value=0, value=2500),
    "insurance_coverage": st.number_input("Insurance Coverage (₹)", min_value=0, value=10000),
    "daily_spent_food": st.number_input("Daily Spent on Food (₹)", min_value=0, value=100),
    "daily_spent_travel": st.number_input("Daily Spent on Travel (₹)", min_value=0, value=50),
    "daily_spent_health": st.number_input("Daily Spent on Health (₹)", min_value=0, value=30),
    "daily_spent_savings": st.number_input("Daily Spent on Savings (₹)", min_value=0, value=20)
}

# Map land ownership and vehicle ownership to numeric values
input_data["land_ownership"] = 1 if input_data["land_ownership"] == "Owned" else 0
input_data["vehicle_owned"] = 1 if input_data["vehicle_owned"] == "Yes" else 0

# Display button to calculate credit score
if st.button("Predict Credit Score"):
    # Get predicted credit score
    predicted_score = predict_credit_score(input_data)
    
    # Display the predicted score
    st.write(f"### Predicted Credit Score: {predicted_score:.2f}")
