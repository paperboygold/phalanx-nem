import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import time
import zipfile
import datetime
import os
import logging
import concurrent.futures
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for the reports
base_url = "https://nemweb.com.au"

# Ensure local data directory exists
data_directory = './data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Regex patterns for target files
target_file_patterns = [
    #r".*PUBLIC_DVD_P5MIN_CONSTRAINTSOLUTION\d+_ALL_.*\.zip$",
    #r".*PUBLIC_DVD_P5MIN_INTERCONNECTORSOLN_ALL_.*\.zip$",
    r".*PUBLIC_DVD_P5MIN_REGIONSOLUTION_ALL_.*\.zip$",
    #r".*PUBLIC_DVD_P5MIN_UNITSOLUTION_ALL_.*\.zip$"
]

# Setup session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount('https://', adapter)
session.mount('http://', adapter)

def fetch_file_links(url):
    response = session.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to access {url}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        href = link.get('href').strip()
        logging.debug(f"Found link: {href}")
        if any(re.match(pattern, href) for pattern in target_file_patterns):
            if href.startswith('/'):
                full_url = f"{base_url}{href}"
            else:
                full_url = href
            links.append(full_url)
            logging.debug(f"Matched and added to download list: {full_url}")
        else:
            logging.debug(f"No match for: {href}")
    return links

def download_file(link, base_directory, attempt=1):
    max_attempts = 5
    backoff_factor = 2  # Exponential backoff factor
    logging.debug(f"Attempting to download file: {link}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = session.get(link, headers=headers)
        logging.debug(f"HTTP response code: {response.status_code}")
        if response.status_code == 200:
            file_name = link.split('/')[-1]
            file_path = os.path.join(base_directory, file_name)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(file_path))
                os.remove(file_path)
            return True, link
        elif response.status_code == 403:
            logging.debug(f"403 Forbidden error for URL: {link}")  # Log only in debug mode
            if attempt < max_attempts:
                sleep_time = backoff_factor ** attempt
                time.sleep(sleep_time)
                return download_file(link, base_directory, attempt + 1)
            else:
                return False, link
        else:
            logging.error(f"Failed to download {link} with status code {response.status_code}")
            if attempt < max_attempts:
                sleep_time = backoff_factor ** attempt
                time.sleep(sleep_time)
                return download_file(link, base_directory, attempt + 1)
            else:
                return False, link
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        logging.warning(f"Attempt {attempt} failed for {link}. Error: {e}")
        if attempt < max_attempts:
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)
            return download_file(link, base_directory, attempt + 1)
        else:
            logging.error(f"Failed to download {link} after {max_attempts} attempts.")
            return False, link

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

def download_files(links, base_directory):
    failed_downloads = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(download_file, link, base_directory): link for link in links}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(links), desc="Downloading files"):
            success, url = future.result()
            if not success:
                failed_downloads.append(url)
    return failed_downloads

def main():
    start_year = 2024
    start_month = 1
    end_year = 2024
    end_month = 1

    # Generate URLs for historical market data
    urls = generate_urls(start_year, start_month, end_year, end_month)
    # Fetch links from each URL
    all_links = []
    for url in urls:
        links = fetch_file_links(url)
        all_links.extend(links)
    
    # Download all files from fetched links
    failed_downloads = download_files(all_links, data_directory)
    if failed_downloads:
        logging.error(f"Failed to download the following files: {failed_downloads}")
    else:
        logging.info("All files downloaded successfully.")

if __name__ == "__main__":
    main()