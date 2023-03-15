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

# Define the model
model_url = f's3://{s3_bucket}/{model_s3_key}' # Combine bucket name, model file name, and relate S3 path to create S3 model URI
sagemaker_role= 'arn:aws:iam::371380984152:role/SageMaker-SMU-IOT' # Role to give SageMaker permission to access AWS services.
entry_point = 'model.sav' # The name of the file that contains the model
py_version = 'py3' # Python version
version = '1.0-1' # Version of the framework or algorithm

model = SKLearnModel(
    model_data=model_url,
    role=sagemaker_role,
    entry_point=entry_point,
    py_version=py_version,
    framework_version=version
)

# Deploy the model
serverless_config = ServerlessInferenceConfig()
serverless_predictor = model.deploy(serverless_inference_config=serverless_config)
