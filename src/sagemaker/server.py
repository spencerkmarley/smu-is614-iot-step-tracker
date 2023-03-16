import boto3
from sagemaker.sklearn import SKLearnModel
from sagemaker import image_uris
from sagemaker.serverless import ServerlessInferenceConfig
import tarfile

# Zip the model up
model_filename = 'model.tar.gz' # the name of the file you want to save your model to
tar = tarfile.open(model_filename, 'w:gz') # create a tar.gz file
tar.add('model.sav') # add the model file to the tar.gz file

# Upload the model to S3
s3_bucket = 'smu-is614-iot-step-tracker' # the name of the S3 bucket
bucket_prefix = 'models' # the folder in the S3 bucket to store the model
model_s3_key = f'{bucket_prefix}/{model_filename}' # the relative S3 path
s3_client = boto3.client('s3') # create an S3 client
s3_client.upload_file(model_filename, s3_bucket, model_s3_key) # upload the model to S3

# Set up the SageMaker client
aws_region = 'ap-southeast-1'
sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
sagemaker_role= 'arn:aws:iam::371380984152:role/SageMaker-SMU-IOT' # Role to give SageMaker permission to access AWS services.

# Define the model
model_name = 'smu-is614-iot-step-tracker' # The name of the model
model_url = f's3://{s3_bucket}/{model_s3_key}' # Combine bucket name, model file name, and relate S3 path to create S3 model URI
entry_point = 'model.sav' # The name of the file that contains the model
py_version = 'py3' # Python version
framework = 'scikit-learn' # The name of the framework
version = '1.0-1' # Version of the framework or algorithm
container = image_uris.retrieve(region=aws_region, framework=framework, version=version) # Get the container image URI

# Create the model
create_model_response = sagemaker_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=sagemaker_role,
    PrimaryContainer={
        'Image': container,
        'ModelDataUrl': model_url
    })

# Deploy the model
endpoint_name = 'smu-is614-iot-step-tracker' # The name of the endpoint
endpoint_config_name = 'smu-is614-iot-step-tracker' # The name of the endpoint configuration
create_endpoint_response = sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
    )
