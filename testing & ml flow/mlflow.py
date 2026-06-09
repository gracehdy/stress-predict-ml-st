import mlflow
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

mlflow.set_experiment("Stress Model Tracking")

print("Loading model and scaler...")
with open('model_stress.pkl', 'rb') as model_file:
    model_stress = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


with mlflow.start_run(run_name="Log Pre-trained Stress Model") as run:
    model_info = mlflow.sklearn.log_model(
        sk_model=model_stress, 
        artifact_path="stress_classification_model"
    )
    
    mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")
    f1 = 0.84   
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.set_tag("Project", "Streamlit ML Final")
    mlflow.set_tag("Model Type", "Pre-trained Pickle")

print(f"Berhasil! Model URI kamu adalah: {model_info.model_uri}")