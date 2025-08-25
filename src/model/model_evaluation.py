import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model_path: str, test_data_path: str, output_metrics_path: str = "metrics.json"):
    """
    Evaluate a trained model on test data and save evaluation metrics.
    
    Parameters:
        model_path (str): Path to the saved model (pickle file).
        test_data_path (str): Path to the test dataset (CSV file).
        output_metrics_path (str): Path where metrics will be saved as JSON.
    """
    
    # Load model
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # Load test data
    test_data = pd.read_csv(test_data_path)

    # Split features and labels
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba)
    }

    # Save metrics
    with open(output_metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return metrics_dict


if __name__ == "__main__":
    # Correct paths (match with model_building.py and dvc.yaml)
    model_file = "models/model.pkl"
    test_file = "data/features/test_bow.csv"
    metrics_file = "reports/metrics.json"

    results = evaluate_model(model_file, test_file, metrics_file)
    print("Evaluation Metrics:", results)

