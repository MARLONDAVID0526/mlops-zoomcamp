from time import sleep
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id="AKIAQAARFNMJG3UJ6EU5", aws_secret_access_key="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"
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="zoomcamp-mlops-2023-marlon", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-block", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()
