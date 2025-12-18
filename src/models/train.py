import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "credit_scoring"
MODEL_NAME = "credit_scoring_model"

# Set MLFlow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(data_path: str):
    """Load and prepare data"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic preprocessing (customize based on your dataset)
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y

def create_confusion_matrix_plot(y_true, y_pred):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save to file
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return 'confusion_matrix.png'

def train_model(X_train, y_train, X_test, y_test, params):
    """Train model and log to MLFlow"""
    
    with mlflow.start_run():
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        print("Training model...")
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Create and log confusion matrix
        cm_path = create_confusion_matrix_plot(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        mlflow.log_artifact('feature_importance.png')
        
        # Log model
        print("Logging model to MLFlow...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )
        
        # Log additional info
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", "credit_scoring")
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Model logged with run_id: {run_id}")
        
        return model, accuracy

def run_experiments():
    """Run multiple experiments with different hyperparameters"""
    
    # Load data
    X, y = load_data('data/processed/credit_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Define hyperparameter configurations to try
    configs = [
        {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        },
        {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        },
        {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 4
        }
    ]
    
    best_model = None
    best_accuracy = 0
    
    for i, params in enumerate(configs, 1):
        print(f"\n{'='*50}")
        print(f"Experiment {i}/{len(configs)}")
        print(f"{'='*50}")
        
        model, accuracy = train_model(X_train, y_train, X_test, y_test, params)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    print(f"\n{'='*50}")
    print(f"Best model accuracy: {best_accuracy:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_experiments()