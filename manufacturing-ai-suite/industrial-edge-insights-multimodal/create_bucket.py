import boto3
url = "http://$HOST_IP:8000"
user = "minioaccesskey added in .env"
password = "miniosecretkey added in .env"
bucket_name = "multimodeldemo"

client= boto3.client(
            "s3",
            endpoint_url=url,
            aws_access_key_id=user,
            aws_secret_access_key=password
)
client.create_bucket(Bucket=bucket_name)
buckets = client.list_buckets()
print("Buckets:", [b["Name"] for b in buckets.get("Buckets", [])])
