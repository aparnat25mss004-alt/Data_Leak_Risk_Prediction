import mlflow
import mlflow.sklearn

# MLflow experiment
mlflow.set_experiment("Data_Leak_Risk_Prediction")

# Start experiment
with mlflow.start_run():

    mlflow.log_param("model_name", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", 0.90)

    print("MLflow tracking completed!")
    print("Experiment logged successfully!")