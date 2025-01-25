import pandas as pd
import random
import numpy as np

# Define the number of rows
num_rows = 100000

# Generate daily wage and ensure logical annual income
daily_wage = np.random.randint(300, 1000, size=num_rows)
annual_income = daily_wage * 330

# Generate mock data
data = {
    "person_id": range(1, num_rows + 1),
    "education_level": random.choices(
        ["Illiterate", "Primary", "Secondary", "Graduate"], k=num_rows
    ),
    "annual_income": annual_income,
    "daily_wage": daily_wage,
    "ration_card_type": random.choices(["APL", "BPL", "Antyodaya", "None"], k=num_rows),
    "household_size": np.random.randint(1, 10, size=num_rows),
    "land_ownership": np.random.uniform(0, 10, size=num_rows).round(2),
    "loan_repayment_history": np.random.randint(0, 20, size=num_rows),
    "savings": np.random.randint(0, 500000, size=num_rows),
    "number_of_loans": np.random.randint(0, 10, size=num_rows),
    "loan_amount": np.random.randint(0, 500000, size=num_rows),
    "community_reliability_rating": np.random.randint(1, 6, size=num_rows),
    "participation_in_MGNREGA": np.random.randint(0, 100, size=num_rows),
    "mobile_phone_cost": np.random.randint(0, 50000, size=num_rows),
    "health_expenditure": np.random.randint(0, 50000, size=num_rows),
    "child_education_expenditure": np.random.randint(0, 50000, size=num_rows),
    "crop_yield": np.random.randint(0, 5000, size=num_rows),
    "vehicle_owned": random.choices([0, 1], weights=[0.7, 0.3], k=num_rows),  # 30% own vehicles
    "utility_bill_payment": np.random.randint(0, 2000, size=num_rows),
    "insurance_coverage": np.random.randint(0, 10000, size=num_rows),
}

# Calculate daily spending categories
daily_spent_food = np.minimum(daily_wage * 0.5, np.random.randint(50, 200, size=num_rows))
daily_spent_travel = np.maximum(0, daily_wage * 0.2 - np.random.randint(0, 50, size=num_rows))
daily_spent_health = np.minimum(daily_wage * 0.1, np.random.randint(20, 150, size=num_rows))
daily_spent_savings = daily_wage - (daily_spent_food + daily_spent_travel + daily_spent_health)

# Ensure no negative savings
daily_spent_savings = np.maximum(daily_spent_savings, 0)

# Add new spending columns
data.update({
    "daily_spent_food": daily_spent_food,
    "daily_spent_travel": daily_spent_travel,
    "daily_spent_health": daily_spent_health,
    "daily_spent_savings": daily_spent_savings,
})

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file_path = "rural_credit_dataset_with_spending.csv"
df.to_csv(csv_file_path, index=False)

print(f"Dataset with {num_rows} rows saved to '{csv_file_path}'.")
