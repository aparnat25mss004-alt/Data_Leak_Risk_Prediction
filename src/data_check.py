import pandas as pd

# Load dataset
df = pd.read_csv("dataset/employee_activity.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Risk level counts
print("\nRisk Level Count:")
print(df["risk_level"].value_counts())