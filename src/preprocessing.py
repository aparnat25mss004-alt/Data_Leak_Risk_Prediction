import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/employee_activity.csv")

# Convert risk level into numerical values
encoder = LabelEncoder()
df["risk_level"] = encoder.fit_transform(df["risk_level"])

# Separate input and output
X = df.drop(["employee_id", "risk_level"], axis=1)
y = df["risk_level"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nRisk level mapping:")
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))