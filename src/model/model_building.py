import os
import pickle
import yaml
import logging
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# ----------------- Logging Setup -----------------
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_building.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ----------------- Load Params -----------------
def load_params(params_path: str) -> dict:
    """Load parameters from params.yaml"""
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        model_params = params.get("model_building", {})
        logger.debug(f"✅ Parameters loaded: {model_params}")
        return model_params
    except FileNotFoundError:
        logger.error("❌ params.yaml not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing params.yaml: {e}")
        raise


# ----------------- Load Data -----------------
def load_training_data(train_path: str):
    """Load training data from CSV"""
    try:
        logger.info(f"📂 Loading training data from {train_path}")
        train_data = pd.read_csv(train_path)

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        logger.info(f"✅ Training data shape: {train_data.shape}")
        return X_train, y_train
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        raise


# ----------------- Train Model -----------------
def train_model(X_train, y_train, params: dict):
    """Train GradientBoosting model"""
    try:
        logger.info("⚡ Training GradientBoostingClassifier model...")
        clf = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3)
        )
        clf.fit(X_train, y_train)
        logger.info("✅ Model training completed")
        return clf
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


# ----------------- Save Model -----------------
def save_model(model, model_path: str):
    """Save model using pickle"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"✅ Model saved at {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise


# ----------------- Main -----------------
def main():
    try:
        params = load_params("params.yaml")
        # ⬅️ Changed file to TF-IDF features
        X_train, y_train = load_training_data("data/features/train_tfidf.csv")
        model = train_model(X_train, y_train, params)
        save_model(model, "models/model.pkl")
        logger.info("🎯 Model building pipeline completed successfully (TF-IDF).")
    except Exception as e:
        logger.critical(f"🔥 Pipeline failed: {e}")


if __name__ == "__main__":
    main()
