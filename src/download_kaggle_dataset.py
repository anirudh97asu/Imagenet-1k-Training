r"""
Complete script to download entire datasets or specific subfolders from Kaggle

Prerequisites:
1. Install kaggle package: pip install kaggle
2. Get your Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - This downloads kaggle.json
   - Place it in: ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<username>\.kaggle\kaggle.json (Windows)
3. Set permissions (Linux/Mac): chmod 600 ~/.kaggle/kaggle.json
"""

import os
import kaggle
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_entire_dataset(dataset_name, download_path='./kaggle_data'):
    """
    Download an entire dataset from Kaggle
    
    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., 'username/dataset-name')
        download_path (str): Local path where dataset will be downloaded
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading entire dataset: {dataset_name}")
    print(f"Download location: {download_path}")
    
    # Download the entire dataset
    api.dataset_download_files(
        dataset_name,
        path=download_path,
        unzip=True  # Automatically unzip the downloaded files
    )
    
    print("Download complete!")
    print(f"Files saved to: {os.path.abspath(download_path)}")


def download_subfolder(dataset_name, subfolder_path, download_path='./kaggle_data'):
    """
    Download a specific subfolder or file from a Kaggle dataset
    
    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., 'username/dataset-name')
        subfolder_path (str): Path to the subfolder/file within the dataset (e.g., 'State' or 'data/train')
        download_path (str): Local path where files will be downloaded
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading subfolder/file: {subfolder_path}")
    print(f"From dataset: {dataset_name}")
    print(f"Download location: {download_path}")
    
    # Download the specific subfolder/file
    api.dataset_download_file(
        dataset_name,
        file_name=subfolder_path,
        path=download_path
    )
    
    # The file will be downloaded as a zip, let's unzip it
    zip_filename = os.path.join(download_path, os.path.basename(subfolder_path) + '.zip')
    
    if os.path.exists(zip_filename):
        print(f"Unzipping {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(zip_filename)  # Remove the zip file after extraction
        print("Extraction complete!")
    
    print(f"Files saved to: {os.path.abspath(download_path)}")


if __name__ == "__main__":
    # Example 1: Download entire dataset
    # To use this script:
    # 1. Comment out the examples above
    # 2. Uncomment and modify one of the options below:
    
    # Option A: Download entire dataset
    download_entire_dataset(r"anishdevedward/loan-approval-dataset", download_path='./data')
    
    # Option B: Download specific subfolder
    # download_subfolder("username/dataset-name", "subfolder/path", download_path='./data')