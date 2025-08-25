import os
import yaml
import pandas as pd
import logging
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------
# Logging setup
# ----------------------------------
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler("feature_engineering.log")
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers once
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ----------------------------------
# Functions
# ----------------------------------

def load_params(params_path: str) -> int:
    """Load max_features from params.yaml"""
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        max_features = params["feature_engineering"]["max_features"]
        logger.debug(f"✅ max_features retrieved: {max_features}")
        return max_features
    except FileNotFoundError:
        logger.error("❌ params.yaml not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"❌ YAML parsing error: {e}")
        raise
    except KeyError:
        logger.error("❌ Missing key: 'feature_engineering -> max_features' in params.yaml")
        raise


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train & test data"""
    try:
        train_data = pd.read_csv("data/processed/train_processed.csv")
        test_data = pd.read_csv("data/processed/test_processed.csv")

        train_data.fillna("", inplace=True)
        test_data.fillna("", inplace=True)

        logger.debug("✅ Preprocessed data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        logger.error(f"❌ Failed to load preprocessed data: {e}")
        raise


def create_bow_features(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
    """Generate Bag of Words features"""
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(train_data["processed"].values)
        X_test_bow = vectorizer.transform(test_data["processed"].values)

        train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
        train_df["label"] = train_data["sentiment"].values

        test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
        test_df["label"] = test_data["sentiment"].values

        logger.debug("✅ Bag of Words features created.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"❌ Error creating BoW features: {e}")
        raise


def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str = "data/features") -> None:
    """Save BoW features to CSV"""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        logger.debug("✅ Features saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save features: {e}")
        raise


def main():
    try:
        max_features = load_params("params.yaml")
        train_data, test_data = load_data()
        train_df, test_df = create_bow_features(train_data, test_data, max_features)
        save_features(train_df, test_df)
        logger.info("✅ Feature engineering pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"🔥 Feature engineering pipeline failed: {e}")


if __name__ == "__main__":
    main()
