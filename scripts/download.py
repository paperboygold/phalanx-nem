import asyncio
from aiohttp_retry import RetryClient, ExponentialRetry
from bs4 import BeautifulSoup
import zipfile
import datetime
import os
import logging
import re
import argparse
from tqdm.asyncio import tqdm
import yaml
import pathlib
from aiolimiter import AsyncLimiter
from aiohttp import ClientError

# Explicitly define the project root assuming the script is in the 'scripts' subdirectory of the project root
project_root = pathlib.Path(__file__).resolve().parent.parent

# Load configuration from file
with open(project_root / "configs/download_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set the data directory from config or default to 'data/raw' relative to the explicitly defined project root
output_path = config.get("output_path", 'data/raw')  # Get output path from config or use default
data_directory = project_root / output_path  # Construct full path relative to project root
data_directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# Configure logging
logging.basicConfig(level=config.get("log_level", "INFO"), format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for the reports
base_url = config["base_url"]

# Regex patterns for target files. Only download region files for now.
target_file_patterns = config["target_file_patterns"]

def configure_logging(level):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

# Async session setup with retry and rate limiting
async def setup_session():
    retry_options = ExponentialRetry(attempts=config["max_attempts"])
    client = RetryClient(retry_options=retry_options)
    limiter = AsyncLimiter(max_rate=config["max_requests_per_second"], time_period=1)
    return client, limiter

async def fetch_file_links(client, limiter, url):
    attempts = 0
    max_attempts = 3  # or fetch from config
    while attempts < max_attempts:
        try:
            async with limiter:
                async with client.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        soup = BeautifulSoup(text, 'html.parser')
                        links = [f"{base_url}{link.get('href').strip()}" if link.get('href').startswith('/') else link.get('href').strip()
                                 for link in soup.find_all('a') if any(re.match(pattern, link.get('href').strip()) for pattern in target_file_patterns)]
                        return links
                    else:
                        logging.error(f"Failed to access {url} with status {response.status}")
        except ClientError as e:
            logging.error(f"Client error when accessing {url}: {e}")
        attempts += 1
        logging.info(f"Retrying {url} ({attempts}/{max_attempts})")
    return []

async def download_file(client, limiter, link, base_directory):
    try:
        async with limiter:
            async with client.get(link) as response:
                if response.status == 200:
                    file_name = link.split('/')[-1]
                    file_path = os.path.join(base_directory, file_name)
                    try:
                        with open(file_path, 'wb') as f:
                            f.write(await response.read())
                    except IOError as e:
                        logging.error(f"Failed to write to file {file_path}: {e}")
                        return False, link
                    if file_path.endswith('.zip'):
                        try:
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(os.path.dirname(file_path))
                            os.remove(file_path)
                        except zipfile.BadZipFile as e:
                            logging.error(f"Zip extraction failed for {file_path}: {e}")
                            return False, link
                    return True, link
                else:
                    logging.error(f"Failed to download {link} with status {response.status}")
                    return False, link
    except Exception as e:
        logging.error(f"Error downloading {link}: {e}")
        return False, link
            
async def download_files(links, base_directory):
    client, limiter = await setup_session()
    tasks = [download_file(client, limiter, link, base_directory) for link in links]
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        results.append(result)
    await client.close()
    return [url for success, url in results if not success]

def generate_urls(start_year, start_month, end_year, end_month):
    base_url = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
    start_date = datetime.date(start_year, start_month, 1)
    end_date = datetime.date(end_year, end_month, 1)
    current_date = start_date

    urls = []

    while current_date <= end_date:
        year = current_date.year
        month = current_date.strftime('%m')
        url = f"{base_url}{year}/MMSDM_{year}_{month}/MMSDM_Historical_Data_SQLLoader/P5MIN_ALL_DATA/"
        urls.append(url)
        if current_date.month == 12:
            current_date = datetime.date(current_date.year + 1, 1, 1)
        else:
            current_date = datetime.date(current_date.year, current_date.month + 1, 1)

    return urls

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download historical market data from AEMO.')
    parser.add_argument('--start-year', type=int, help='Start year')
    parser.add_argument('--start-month', type=int, help='Start month')
    parser.add_argument('--end-year', type=int, help='End year')
    parser.add_argument('--end-month', type=int, help='End month')
    parser.add_argument('--data-dir', type=str, help='Data directory override')
    parser.add_argument('--log-level', type=str, help='Log level')
    return parser.parse_args()

def setup_directories(base_directory):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

async def process_downloads(start_year, start_month, end_year, end_month, base_directory):
    urls = generate_urls(start_year, start_month, end_year, end_month)
    client, limiter = await setup_session()
    all_links = []
    for url in urls:
        links = await fetch_file_links(client, limiter, url)
        all_links.extend(links)
    failed_downloads = await download_files(all_links, base_directory)
    if failed_downloads:
        logging.error(f"Failed to download the following files: {failed_downloads}")
    else:
        logging.info("All files downloaded successfully.")
    await client.close()

async def main():
    args = parse_arguments()
    configure_logging(args.log_level or config.get("log_level", "INFO"))
    base_directory = args.data_dir or data_directory
    setup_directories(base_directory)
    await process_downloads(args.start_year or config.get("start_year"),
                            args.start_month or config.get("start_month"),
                            args.end_year or config.get("end_year"),
                            args.end_month or config.get("end_month"),
                            base_directory)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")