
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("dataset/employee_activity.csv")

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

# Models
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
}

# MLflow experiment
mlflow.set_experiment("Data_Leak_Risk_Prediction")

results = []

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train model
        model.fit(X_train, y_train)

        # Prediction
        predictions = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        # Log metrics
        mlflow.log_param("model_name", model_name)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"\n{model_name} completed")
        print(f"Accuracy : {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall   : {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")

# Comparison table
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# Best model
best_model = results_df.sort_values(
    by="F1 Score", ascending=False
).iloc[0]

print("\n===== BEST MODEL =====")
print("Model:", best_model["Model"])
print("F1 Score:", f'{best_model["F1 Score"]:.2%}')
