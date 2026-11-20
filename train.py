import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("dataset/employee_activity.csv")

# Encode target
encoder = LabelEncoder()
df["risk_level"] = encoder.fit_transform(df["risk_level"])

# Features and target
X = df.drop(["employee_id", "risk_level"], axis=1)
y = df["risk_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

# MLflow tracking
mlflow.set_experiment("Data_Leak_Risk_Prediction")

with mlflow.start_run():

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
    model,
    name="risk_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

# Save model locally
joblib.dump(model, "models/risk_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("\nModel Training Completed!")
print("----------------------")
print(f"Accuracy  : {accuracy:.2%}")
print(f"Precision : {precision:.2%}")
print(f"Recall    : {recall:.2%}")
print(f"F1 Score : {f1:.2%}")

print("\nMetrics successfully logged to MLflow!")
print("Model saved in models folder!")