import boto3
import os
from dotenv import load_dotenv

# Get the root directory of the project dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load .env from the project root
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)


# AWS S3 config
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
region_name = os.environ.get("AWS_REGION", "us-east-1")
local_root = os.environ.get("LOCAL_DATA_PATH")

if not local_root:
    raise ValueError("LOCAL_DATA_PATH not set. Check your .env file!")

bucket_name = "urban-waste-cnn"

# Initialize S3 client
s3 = boto3.client('s3',
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=region_name)

# Loop through each subfolder (images0, images1, ...)
for folder_name in os.listdir(local_root):
    folder_path = os.path.join(local_root, folder_name)
    if os.path.isdir(folder_path):
        print(f"Uploading folder: {folder_name}")
        for file_name in os.listdir(folder_path):
            local_file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(local_file_path):
                # Create S3 key with folder prefix
                s3_key = f"{folder_name}/{file_name}"
                s3.upload_file(local_file_path, bucket_name, s3_key)
                print(f"Uploaded {local_file_path} → s3://{bucket_name}/{s3_key}")

for json_file in ["training.json", "testing.json"]:
    local_path = os.path.join(local_root, json_file)
    if os.path.isfile(local_path):
        s3.upload_file(local_path, bucket_name, json_file)
        print(f"Uploaded {json_file} → s3://{bucket_name}/{json_file}")
