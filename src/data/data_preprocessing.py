import os
import re
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------
# Logging setup
# ----------------------------------
logger = logging.getLogger("data-preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("data_preprocessing.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ----------------------------------
# NLTK Downloads
# ----------------------------------
nltk.download("wordnet")
nltk.download("stopwords")

# ----------------------------------
# Text Cleaning Functions
# ----------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def lemmatization(text: str) -> str:
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text: str) -> str:
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text: str) -> str:
    return "".join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    return text.lower()

def removing_punctuations(text: str) -> str:
    text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~،؛؟]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def removing_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["processed"] = df["content"].astype(str)
        df["processed"] = df["processed"].apply(lower_case)
        df["processed"] = df["processed"].apply(remove_stop_words)
        df["processed"] = df["processed"].apply(removing_numbers)
        df["processed"] = df["processed"].apply(removing_punctuations)
        df["processed"] = df["processed"].apply(removing_urls)
        df["processed"] = df["processed"].apply(lemmatization)
        logger.debug("✅ Text normalization complete")
        return df[["processed", "sentiment"]]
    except KeyError as e:
        logger.error(f"❌ Missing required column: {e}")
        raise

# ----------------------------------
# Main Preprocessing Pipeline
# ----------------------------------
def preprocess_data(train_path: str, test_path: str, output_dir: str):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info("✅ Raw data loaded successfully")

        # Check required column
        if "content" not in train_data.columns or "content" not in test_data.columns:
            raise KeyError("❌ 'content' column not found in input data")

        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        os.makedirs(output_dir, exist_ok=True)
        train_processed.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)

        logger.info("✅ Data preprocessing complete and saved")
    except Exception as e:
        logger.critical(f"🔥 Preprocessing pipeline failed: {e}")
        raise

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    preprocess_data(
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv",
        output_dir="data/processed"
    )
