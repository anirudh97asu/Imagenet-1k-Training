#!/usr/bin/env python3
import os
import subprocess
import boto3
from pathlib import Path
from tqdm import tqdm
import sys

# Configuration
KAGGLE_COMPETITION = 'imagenet-object-localization-challenge'
DOWNLOAD_DIR = '/home/ec2-user/imagenet'
S3_BUCKET = 'my-imagenet-bucket'  # CHANGE THIS to your bucket name
S3_REGION = 'us-east-1'
S3_STORAGE_CLASS = 'GLACIER_IR'  # Glacier Instant Retrieval - best balance

# Create download directory
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.chdir(DOWNLOAD_DIR)

print("=" * 60)
print("Step 1: Downloading ImageNet from Kaggle")
print("=" * 60)

try:
    # Download from Kaggle with live output
    process = subprocess.Popen(
        ['kaggle', 'competitions', 'download', '-c', KAGGLE_COMPETITION],
        cwd=DOWNLOAD_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Show output as it happens
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode == 0:
        print("✓ Download completed")
    else:
        print("✗ Download failed with return code:", process.returncode)
        exit(1)
        
except Exception as e:
    print(f"✗ Error during download: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Step 2: Extracting files")
print("=" * 60)

try:
    # Find all zip files
    zip_files = list(Path(DOWNLOAD_DIR).glob('*.zip'))
    
    for zip_file in tqdm(zip_files, desc="Extracting ZIP files", unit="file"):
        subprocess.run(['unzip', '-q', str(zip_file)], cwd=DOWNLOAD_DIR, check=True)
    
    # Extract tar files if any
    tar_files = list(Path(DOWNLOAD_DIR).glob('*.tar'))
    
    for tar_file in tqdm(tar_files, desc="Extracting TAR files", unit="file"):
        subprocess.run(['tar', '-xf', str(tar_file)], cwd=DOWNLOAD_DIR, check=True)
    
    print("✓ Extraction complete")
        
except Exception as e:
    print(f"✗ Extraction failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Step 3: Preparing files for upload")
print("=" * 60)

try:
    # Find the ILSVRC folder
    ilsvrc_path = None
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        if 'Data' in dirs:
            data_path = os.path.join(root, 'Data', 'CLS-LOC')
            if os.path.exists(data_path):
                ilsvrc_path = data_path
                break
    
    if not ilsvrc_path:
        print("✗ Could not find ILSVRC/Data/CLS-LOC folder")
        print(f"Contents of {DOWNLOAD_DIR}:")
        for item in os.listdir(DOWNLOAD_DIR):
            print(f"  - {item}")
        exit(1)
    
    print(f"✓ Found data folder: {ilsvrc_path}")
    
    # Count total files
    total_files = 0
    for root, dirs, files in os.walk(ilsvrc_path):
        total_files += len(files)
    
    print(f"✓ Found {total_files} files to upload")
    
except Exception as e:
    print(f"✗ Preparation failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Step 4: Uploading to S3")
print("=" * 60)

try:
    s3_client = boto3.client('s3', region_name=S3_REGION)
    
    print(f"Uploading to s3://{S3_BUCKET}/imagenet/")
    
    # Create progress bar
    pbar = tqdm(total=total_files, desc="Uploading files", unit="file")
    
    # Upload all files
    for root, dirs, files in os.walk(ilsvrc_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, ilsvrc_path)
            s3_key = f'imagenet/{relative_path}'
            
            try:
                s3_client.upload_file(
                    local_file,
                    S3_BUCKET,
                    s3_key,
                    ExtraArgs={'StorageClass': S3_STORAGE_CLASS}
                )
                pbar.update(1)
                    
            except Exception as e:
                pbar.write(f"✗ Failed to upload {s3_key}: {e}")
                pbar.update(1)
    
    pbar.close()
    print("✓ Upload complete!")
    
    # Verify upload
    print("\nVerifying S3 upload...")
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix='imagenet/')
    uploaded_count = response.get('KeyCount', 0)
    print(f"✓ Verified: {uploaded_count} files in S3")
    
except Exception as e:
    print(f"✗ S3 upload failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ SUCCESS! ImageNet is now in S3")
print("=" * 60)
print(f"Bucket: s3://{S3_BUCKET}/imagenet/")
print(f"Files: {uploaded_count}")
print("=" * 60)