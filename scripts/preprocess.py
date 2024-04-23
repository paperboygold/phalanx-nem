import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pathlib
import logging
import yaml
from logging.handlers import RotatingFileHandler

# Setup logging
def setup_logging():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    log_directory = project_root / 'logs'
    log_directory.mkdir(exist_ok=True)
    log_file = log_directory / 'preprocess.log'

    # Configure logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            RotatingFileHandler(log_file, maxBytes=10485760, backupCount=3)
                        ])

setup_logging()

# Explicitly define the project root assuming the script is in the 'scripts' subdirectory of the project root
project_root = pathlib.Path(__file__).resolve().parent.parent

# Set the data directories relative to the project root
raw_data_directory = project_root / 'data/raw'
processed_data_path = project_root / 'data/processed/processed_data.csv'

# Ensure the directories exist
raw_data_directory.mkdir(parents=True, exist_ok=True)
processed_data_path.parent.mkdir(parents=True, exist_ok=True)

# Load configuration from file using pathlib
config_path = project_root / "configs/columns_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def load_data(directory):
    logging.info("Loading data from directory: %s", directory)
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.CSV')]
    # Ensure the header is correctly recognized by specifying header=0
    data = pd.concat((pd.read_csv(file, header=1, parse_dates=['RUN_DATETIME', 'INTERVAL_DATETIME', 'LASTCHANGED']) for file in files), ignore_index=True)
    logging.info("Data loaded successfully with %d records", len(data))
    return data

def parse_dates(data):
    logging.info("Parsing date columns")
    date_cols = ['RUN_DATETIME', 'INTERVAL_DATETIME', 'LASTCHANGED']
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    return data

def add_time_features(data):
    logging.info("Adding time features")
    data['hour'] = data['RUN_DATETIME'].dt.hour
    data['day_of_week'] = data['RUN_DATETIME'].dt.dayofweek
    data['month'] = data['RUN_DATETIME'].dt.month
    data['year'] = data['RUN_DATETIME'].dt.year
    return data

def encode_categorical(data):
    logging.info("Encoding categorical columns")
    categorical_cols = data.select_dtypes(include=['object']).columns
    high_cardinality_cols = ['REGIONID']
    low_cardinality_cols = [col for col in categorical_cols if col not in high_cardinality_cols]
    
    data = pd.get_dummies(data, columns=low_cardinality_cols)
    for col in high_cardinality_cols:
        data[col] = data[col].astype('category').cat.codes
    return data

def keep_columns(data, columns_to_keep):
    logging.info("Keeping only specified columns: %s", columns_to_keep)
    # Keep only the specified columns
    data = data[columns_to_keep]
    logging.info("Unspecified columns dropped successfully")
    return data

def handle_outliers(data):
    logging.info("Handling outliers")
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        mean, std = data[col].mean(), data[col].std()
        upper_limit = mean + 3 * std
        lower_limit = mean - 3 * std
        data[col] = np.clip(data[col], lower_limit, upper_limit)
    return data

def normalize_data(data):
    logging.info("Normalizing data")
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data

def handle_missing_values(data):
    logging.info("Handling missing values")
    data = data.fillna(data.mean())
    return data

def save_data(data, filepath):
    logging.info("Saving data to %s", filepath)
    data.to_csv(filepath, index=False)
    logging.info("Data saved successfully")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    config = load_config(config_path)
    columns_to_keep = config['columns_to_keep']
    logging.info("Starting preprocessing pipeline")
    data = load_data(raw_data_directory)
    data = parse_dates(data)
    data = add_time_features(data)
    data = keep_columns(data, columns_to_keep)
    data = handle_outliers(data)
    data = normalize_data(data)
    data = encode_categorical(data)
    save_data(data, processed_data_path)
    logging.info("Preprocessing pipeline completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred: %s", e)

