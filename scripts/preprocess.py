import numpy as np
from dask_ml.preprocessing import StandardScaler
import os
import pathlib
import logging
import yaml
import dask.dataframe as dd
from logging.handlers import RotatingFileHandler

# Setup logging
def setup_logging():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    log_directory = project_root / 'logs'
    log_directory.mkdir(exist_ok=True)
    log_file = log_directory / 'preprocess.log'

    # Configure logging
    logging.basicConfig(level=logging.INFO,
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

def load_data(directory, date_columns):
    try:
        logging.info("Loading data from directory: %s", directory)
        files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.CSV')]
        logging.info("Number of files to process: %d", len(files))  # Log the number of files
        data = dd.read_csv(files, header=1, assume_missing=True)
        logging.info("Data loaded successfully")
    except Exception as e:
        logging.error("Failed to load data from directory %s: %s", directory, e, exc_info=True)
        raise
    return data

def parse_dates(data, date_columns):
    logging.info("Parsing date columns")
    existing_columns = data.columns
    date_cols_to_parse = [col for col in date_columns if col in existing_columns]
    if not date_cols_to_parse:
        logging.warning("None of the specified date columns are present in the data.")
    for col in date_cols_to_parse:
        data[col] = dd.to_datetime(data[col], errors='coerce')
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
    
    # Convert low cardinality columns to categorical dtype and encode
    for col in low_cardinality_cols:
        data[col] = data[col].astype('category').cat.as_known()
        data[col] = data[col].cat.codes
    
    # Handle high cardinality columns separately if needed
    for col in high_cardinality_cols:
        data[col] = data[col].astype('category').cat.as_known()
        data[col] = data[col].cat.codes
    
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

from dask.diagnostics import ProgressBar

def normalize_data(data):
    logging.info("Normalizing data")
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns    
    # Apply normalization
    transformed_data = scaler.fit_transform(data[numerical_cols])
    # Use assign to create a new DataFrame with the normalized columns
    data = data.assign(**{col: transformed_data[col] for col in numerical_cols})    
    return data

def handle_missing_values(data):
    logging.info("Handling missing values")
    # Select only numeric columns for filling missing values
    numeric_cols = data.select_dtypes(include=['number']).columns
    # Compute the mean for each numeric column and use fillna in a way that Dask can handle
    for col in numeric_cols:
        mean = data[col].mean().compute()  # Ensure computation of mean
        data[col] = data[col].fillna(mean)  # Apply fillna on a column-by-column basis
    return data

def save_data(data, filepath):
    try:
        parquet_path = filepath.with_suffix('.parquet')
        logging.info("Saving data to %s", parquet_path)
        data.to_parquet(parquet_path)  # Removed index=False
        logging.info("Data saved successfully in Parquet format")
    except Exception as e:
        logging.error("Failed to save data to %s: %s", parquet_path, e, exc_info=True)
        raise

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    try:
        config = load_config(config_path)
        columns_to_keep = config['columns_to_keep']
        date_columns = config['date_columns']
        logging.info("Starting preprocessing pipeline")
        data = load_data(raw_data_directory, date_columns)
        data = parse_dates(data, date_columns)
        data = add_time_features(data)
        data = keep_columns(data, columns_to_keep)
        data = handle_missing_values(data)
        data = data.persist()
        data = handle_outliers(data)
        data = data.persist()
        data = normalize_data(data)
        data = encode_categorical(data)
        save_data(data, processed_data_path)
        logging.info("Preprocessing pipeline completed")
    except Exception as e:
        logging.error("An error occurred in the preprocessing pipeline: %s", e, exc_info=True)
    data = dd.read_parquet('/home/socol/Workspace/phalanx-nem/data/processed/processed_data.parquet')
    num_rows = data.shape[0].compute()
    print(f"Number of rows in the dataset: {num_rows}")

if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except Exception as e:
        logging.error("An error occurred when running the script: %s", e, exc_info=True)