import boto3

from src.config import KEYS

# Specify your AWS Region
aws_region=KEYS.AWS_DEFAULT_REGION

# Create a low-level SageMaker service client.
sagemaker_client = boto3.client('sagemaker', region_name=aws_region)

# Role to give SageMaker permission to access AWS services.
sagemaker_role= "arn:aws:iam::371380984152:role/SageMaker-SMU-IOT"