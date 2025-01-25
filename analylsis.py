import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Start timer to ensure script runs within 1 minute
start_time = time.time()

# Load the dataset
dataset_path = "rural_credit_dataset_with_final_scores.csv"
df = pd.read_csv(dataset_path, on_bad_lines='skip')

print(df.size)

# Feature selection: selecting relevant columns
selected_columns = [
    "annual_income", "daily_wage", "household_size", "land_ownership",
    "loan_repayment_history", "savings", "number_of_loans", "loan_amount",
    "community_reliability_rating", "participation_in_MGNREGA", "mobile_phone_cost",
    "health_expenditure", "child_education_expenditure", "crop_yield", "vehicle_owned",
    "utility_bill_payment", "insurance_coverage", "daily_spent_food", "daily_spent_travel",
    "daily_spent_health", "daily_spent_savings"
]

# Define features (X) and target (y)
X = df[selected_columns]
y = df["credit_score"]

# Handle categorical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Check for NaN or infinity in target variable (y)
if pd.isna(y).any() or np.isinf(y).any():
    print("y contains NaN or infinity values.")
    y = y.replace([np.inf, -np.inf], 0).fillna(0)  # Replace NaN and infinity with 0

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for NaN or infinity in y_train and y_test
if pd.isna(y_train).any() or np.isinf(y_train).any():
    print("y_train contains NaN or infinity values.")
    y_train = y_train.replace([np.inf, -np.inf], 0).fillna(0)

if pd.isna(y_test).any() or np.isinf(y_test).any():
    print("y_test contains NaN or infinity values.")
    y_test = y_test.replace([np.inf, -np.inf], 0).fillna(0)

# Convert datasets to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost parameters optimized for GPU and fast execution
xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "tree_method": "gpu_hist",  # Enables GPU acceleration
    "predictor": "gpu_predictor",  # Ensures GPU usage for prediction
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 1,
    "random_state": 42,
}

# Train the XGBoost model
model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=50,  # Faster convergence by limiting boosting rounds
)

# Make predictions
y_pred = model.predict(dtest)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained model to a pickle file
model_file = "credit_score_xgb_model.pkl"
with open(model_file, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved to '{model_file}'")

# Plot feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='weight', max_num_features=10, title="Top 10 Important Features")
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted Credit Scores")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# End timer and print runtime
end_time = time.time()
print(f"Script completed in {end_time - start_time:.2f} seconds.")
