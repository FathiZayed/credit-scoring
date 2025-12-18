# scripts/export_model.py

import os
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI from environment variable
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Set MLflow tracking URI to: {tracking_uri}")
else:
    print("Warning: MLFLOW_TRACKING_URI not set. Using local tracking or failing.")

client = MlflowClient()

# --- Configuration ---
MODEL_NAME = "credit_scoring_model"
MODEL_STAGE = "Production"  # Or "Staging"
EXPORT_PATH = "models/exported_model"
# --- End Configuration ---

try:
    # Find the latest model version in the specified stage
    model_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
    print(f"Found model: {model_version.name}, version {model_version.version} in stage '{model_version.current_stage}'")

    # Download the model artifacts to the export path
    print(f"Downloading model artifacts to '{EXPORT_PATH}'...")
    os.makedirs(EXPORT_PATH, exist_ok=True)
    client.download_artifacts(model_version.run_id, "model", dst_path=EXPORT_PATH)
    
    print("✅ Model export successful.")

except IndexError:
    print(f"❌ No model named '{MODEL_NAME}' found in stage '{MODEL_STAGE}'.")
    print(f"Please ensure you have registered and promoted a model in MLflow.")
except Exception as e:
    print(f"❌ An error occurred during model export: {e}")