
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, HfFolder
from sklearn.model_selection import GridSearchCV # Import GridSearchCV

# Define constants
HF_DATASET_REPO_ID = "pjhansi2404/Cust_Tourpacks"
HF_MODEL_REPO_ID = "pjhansi2404/Cust_Tourpacks_Model"

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlruns.db") # Or a remote tracking server
mlflow.set_experiment("Wellness_Tourism_Package_Prediction")

# Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

def load_data_from_hf(filename):
    dataset_path = f"hf://datasets/{HF_DATASET_REPO_ID}/{filename}"
    return pd.read_csv(dataset_path)


if __name__ == "__main__":
    print("Loading data from Hugging Face...")
    Xtrain = load_data_from_hf("Xtrain.csv")
    Xtest = load_data_from_hf("Xtest.csv")
    ytrain = load_data_from_hf("ytrain.csv")
    ytest = load_data_from_hf("ytest.csv")
    print("Data loaded successfully.")

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Define parameter grid for GridSearchCV
        param_grid = {
            'solver': ['liblinear', 'lbfgs'],
            'C': [0.1, 1.0, 10.0],
            'random_state': [42]
        }

        # Initialize Logistic Regression model
        lr = LogisticRegression()

        # Initialize GridSearchCV
        print("Performing hyperparameter tuning with GridSearchCV...")
        grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(Xtrain, ytrain.values.ravel())

        # Get the best model and best parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print("Hyperparameter tuning complete. Best parameters found:", best_params)

        # Log best parameters to MLflow
        mlflow.log_params(best_params)

        # Make predictions with the best model
        y_pred = best_model.predict(Xtest)

        # Evaluate the best model
        accuracy = accuracy_score(ytest, y_pred)
        precision = precision_score(ytest, y_pred)
        recall = recall_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Log the best model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="logistic_regression_model",
            registered_model_name="LogisticRegressionModel"
        )
        print("Best model logged and registered with MLflow.")

        # Save best model locally
        local_model_path = "logistic_regression_model.pkl"
        mlflow.sklearn.save_model(best_model, local_model_path)
        print(f"Best model saved locally to {local_model_path}")

        # Upload best model to Hugging Face
        try:
            api.create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True)
            api.upload_file(
                path_or_fileobj=local_model_path,
                path_in_repo="logistic_regression_model.pkl",
                repo_id=HF_MODEL_REPO_ID,
                repo_type="model",
            )
            print(f"Best model uploaded to Hugging Face Hub: {HF_MODEL_REPO_ID}")
        except Exception as e:
            print(f"Error uploading model to Hugging Face: {e}")
