import pandas as pd
import numpy as np

# Load dataset
dataset_path = "rural_credit_dataset_with_spending.csv"
df = pd.read_csv(dataset_path)

# Function to calculate credit score
def calculate_credit_score(row):
    score = 300  # Start with a base score

    # Annual income contribution
    if row["annual_income"] > 300000:
        score += 100
    elif row["annual_income"] > 150000:
        score += 75
    else:
        score += 50

    # Loan repayment history (fewer defaults better)
    score += max(100 - row["loan_repayment_history"] * 5, 0)

    # Savings contribution
    if row["savings"] > 200000:
        score += 100
    elif row["savings"] > 100000:
        score += 75
    else:
        score += 50

    # Daily spending analysis
    daily_spent = row["daily_spent_food"] + row["daily_spent_travel"] + row["daily_spent_health"] + row["daily_spent_savings"]
    if daily_spent > row["daily_wage"]:
        score -= 50  # Overspending is penalized

    # Food spending
    food_ratio = row["daily_spent_food"] / row["daily_wage"]
    if 0.3 <= food_ratio <= 0.5:
        score += 50
    else:
        score -= 20

    # Travel spending
    travel_ratio = row["daily_spent_travel"] / row["daily_wage"]
    if 0.05 <= travel_ratio <= 0.2:
        score += 30
    else:
        score -= 20

    # Savings
    savings_ratio = row["daily_spent_savings"] / row["daily_wage"]
    if 0.15 <= savings_ratio <= 0.3:
        score += 50
    elif savings_ratio > 0.5:
        score -= 20

    # Health spending
    health_ratio = row["daily_spent_health"] / row["daily_wage"]
    if health_ratio <= 0.1:
        score += 30
    else:
        score -= 20

    # Ration card type
    if row["ration_card_type"] == "APL":
        score += 50
    elif row["ration_card_type"] == "BPL":
        score += 30
    elif row["ration_card_type"] == "Antyodaya":
        score += 10

    # Community reliability score
    score += row["community_reliability_rating"] * 20

    # Health expenditure (absolute amount)
    if row["health_expenditure"] / row["annual_income"] <= 0.05:
        score += 30
    else:
        score -= 30

    # Child education expenses
    if row["child_education_expenditure"] > 20000:
        score += 50

    # Insurance coverage
    if row["insurance_coverage"] > 5000:
        score += 30

    # MGNREGA participation (higher indicates financial struggle)
    if row["participation_in_MGNREGA"] > 50:
        score -= 50

    # Vehicle ownership
    if row["vehicle_owned"] >= 1:
        score += 50

    # Cap and normalize score
    return min(max(score, 300), 850)  # Credit scores typically range from 300 to 850

# Apply credit score calculation
df["credit_score"] = df.apply(calculate_credit_score, axis=1)

# Save the updated dataset
output_path = "rural_credit_dataset_with_final_scores.csv"
df.to_csv(output_path, index=False)

print(f"Credit scores updated with spending distribution and dataset saved to '{output_path}'.")
