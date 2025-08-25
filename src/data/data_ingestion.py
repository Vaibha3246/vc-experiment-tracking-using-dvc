import numpy as np
import pandas as pd
import yaml
import os
import logging
import sys
from sklearn.model_selection import train_test_split

# ----------------------------------
# Logging setup (UTF-8 safe)
# ----------------------------------
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Console handler with UTF-8
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# File handler with UTF-8
file_handler = logging.FileHandler('data_ingestion.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers only once
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ----------------------------------
# Functions
# ----------------------------------

def load_params(params_path: str) -> float:
    """Load test_size parameter from YAML."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug(f"✅ test_size retrieved: {test_size}")
        return test_size
    except FileNotFoundError:
        logger.error('❌ params.yaml not found.')
        raise
    except yaml.YAMLError as e:
        logger.error(f'❌ YAML parsing error: {e}')
        raise
    except KeyError:
        logger.error("❌ Missing key: 'data_ingestion -> test_size' in params.yaml")
        raise


def load_data(url: str) -> pd.DataFrame:
    """Load dataset from URL."""
    try:
        df = pd.read_csv(url)
        logger.debug("✅ Data loaded from URL.")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data from URL: {e}")
        raise


def processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter data for sentiment analysis."""
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.debug("✅ Data cleaned and filtered.")
        return final_df
    except KeyError as e:
        logger.error(f"❌ Expected column missing: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the split data into CSV files."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.debug("✅ Data saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save data: {e}")
        raise


def main():
    try:
        test_size = load_params('params.yaml')
        df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = processed_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='data/raw')
        logger.info("✅ Data pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"🔥 Pipeline failed: {e}")


if __name__ == '__main__':
    main()
