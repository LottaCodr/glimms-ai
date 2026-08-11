import boto3, os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_s3_client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

def fetch_image_bytes(key: str) -> bytes:
    s3 = get_s3_client()
    res = s3.get_object(Bucket=os.getenv("S3_BUCKET", "glimms-images"), Key=key)
    return res["Body"].read()

def upload_image_bytes(key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    s3 = get_s3_client()
    s3.put_object(Bucket=os.getenv("S3_BUCKET", "glimms-images"), Key=key, Body=data, ContentType=content_type)
    return f"https://{os.getenv('S3_BUCKET', 'glimms-images')}.s3.amazonaws.com/{key}"
