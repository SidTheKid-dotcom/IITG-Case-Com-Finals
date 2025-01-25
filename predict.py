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

# Static input data for prediction
input_data = {
    "annual_income": 500000.0,
    "daily_wage": 300.0,
    "household_size": 5,
    "land_ownership": 1,  # 1 for own, 0 for rented
    "loan_repayment_history": 8,
    "savings": 20000.0,
    "number_of_loans": 2,
    "loan_amount": 100000.0,
    "community_reliability_rating": 4,
    "participation_in_MGNREGA": 75.0,
    "mobile_phone_cost": 500.0,
    "health_expenditure": 1500.0,
    "child_education_expenditure": 2000.0,
    "crop_yield": 1500.0,
    "vehicle_owned": 1,  # 1 for yes, 0 for no
    "utility_bill_payment": 2500.0,
    "insurance_coverage": 10000.0,
    "daily_spent_food": 100.0,
    "daily_spent_travel": 50.0,
    "daily_spent_health": 30.0,
    "daily_spent_savings": 20.0
}

# Get predicted credit score
predicted_score = predict_credit_score(input_data)

# Output the predicted credit score
print(f"Predicted Credit Score: {predicted_score}")
