import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load("models/risk_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Sample employee data
employee = pd.DataFrame([{
    "files_accessed": 85,
    "download_mb": 2400,
    "sensitive_files_accessed": 8,
    "external_sharing": 1,
    "usb_transfer": 1,
    "after_hours_activity": 1,
    "failed_access_attempts": 5
}])

# Predict risk
prediction = model.predict(employee)

# Convert prediction to risk label
risk_label = encoder.inverse_transform(prediction)

print("Employee Risk Prediction:", risk_label[0])