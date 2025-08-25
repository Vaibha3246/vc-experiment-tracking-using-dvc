# src/models/model_evaluation.py

import os
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model_path: str, test_data_path: str, output_metrics_path: str = "reports/metrics.json"):
    """
    Evaluate the trained model on test data and save evaluation metrics.
    """
    
    # -------- Load Model --------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # -------- Load Test Data --------
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

    test_data = pd.read_csv(test_data_path)

    # Split features & labels
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # -------- Predictions --------
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # -------- Metrics --------
    metrics_dict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_pred_proba))
    }

    # -------- Save Metrics --------
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return metrics_dict


if __name__ == "__main__":
    model_file = "models/model.pkl"
    test_file = "data/features/test_tfidf.csv"   # ✅ match with feature_engineering.py
    metrics_file = "reports/metrics.json"

    results = evaluate_model(model_file, test_file, metrics_file)
    print("✅ Evaluation Completed. Metrics saved to:", metrics_file)
    print("📊 Evaluation Metrics:", results)
