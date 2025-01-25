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

    # Check for missing columns and fill them with default values (like 'None' or 'Illiterate')
    required_columns = ["ration_card_type", "education_level"]
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 'None'  # Fill missing categorical columns with a default value

    # One-hot encoding for categorical variables
    input_df = pd.get_dummies(input_df, columns=["ration_card_type", "education_level"], drop_first=True)

    # Predict the credit score
    predicted_score = model.predict(input_df)[0]
    return predicted_score

# Example input data (replace with real data)
input_data = {
    "annual_income": 330000,
    "daily_wage": 300,
    "household_size": 5,
    "land_ownership": 2,
    "loan_repayment_history": 5,
    "savings": 50000,
    "number_of_loans": 2,
    "loan_amount": 20000,
    "community_reliability_rating": 4,
    "participation_in_MGNREGA": 80,
    "mobile_phone_cost": 3000,
    "health_expenditure": 10000,
    "child_education_expenditure": 15000,
    "crop_yield": 3000,
    "vehicle_owned": 1,
    "utility_bill_payment": 1500,
    "insurance_coverage": 5000,
    "daily_spent_food": 50,
    "daily_spent_travel": 30,
    "daily_spent_health": 10,
    "daily_spent_savings": 20
}

# Get predicted credit score
predicted_score = predict_credit_score(input_data)

# Output the predicted credit score
print(f"Predicted Credit Score: {predicted_score}")
