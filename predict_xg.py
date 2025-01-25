import pandas as pd
import xgboost as xgb
import pickle

# Load the trained model
model_file = "credit_score_xgb_model.pkl"
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Define a sample static input (you can modify these values as needed)
sample_input = {
    "annual_income": 50000,
    "daily_wage": 200,
    "household_size": 5,
    "land_ownership": 1,
    "loan_repayment_history": 0,
    "savings": 1000,
    "number_of_loans": 2,
    "loan_amount": 15000,
    "community_reliability_rating": 4,
    "participation_in_MGNREGA": 1,
    "mobile_phone_cost": 50,
    "health_expenditure": 200,
    "child_education_expenditure": 150,
    "crop_yield": 1200,
    "vehicle_owned": 1,
    "utility_bill_payment": 300,
    "insurance_coverage": 1,
    "daily_spent_food": 50,
    "daily_spent_travel": 20,
    "daily_spent_health": 30,
    "daily_spent_savings": 10
}

# Convert the sample input into a pandas DataFrame
input_df = pd.DataFrame([sample_input])

# Handle categorical variables (one-hot encoding)
input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure the input features match the model's training features (handle missing columns)
model_features = model.feature_names  # Features used during model training
missing_columns = set(model_features) - set(input_df.columns)

for col in missing_columns:
    input_df[col] = 0  # Add missing columns with default value 0

input_df = input_df[model_features]  # Reorder the columns to match the model

# Convert the input DataFrame to DMatrix for prediction
dinput = xgb.DMatrix(input_df)

# Predict the credit score using the trained model
predicted_credit_score = model.predict(dinput)

# Display the predicted credit score
print(f"Predicted Credit Score: {predicted_credit_score[0]:.2f}")
