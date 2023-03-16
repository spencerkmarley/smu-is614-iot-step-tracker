import boto3
import itertools
import numpy as np
import pandas as pd
import os
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# S3 prefix
prefix = "smu-iot"

# Establish a SageMaker session
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::371380984152:role/SageMaker-SMU-IOT'

os.makedirs("./data", exist_ok=True)

# Download the Iris dataset
s3_client = boto3.client("s3")
s3_client.download_file(
    f"sagemaker-sample-files", "datasets/tabular/iris/iris.data", "./data/iris.csv"
)
df_iris = pd.read_csv("./data/iris.csv", header=None)
df_iris[4] = df_iris[4].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
iris = df_iris[[4, 0, 1, 2, 3]].to_numpy()
np.savetxt("./data/iris.csv", iris, delimiter=",", fmt="%1.1f, %1.3f, %1.3f, %1.3f, %1.3f")

# Upload the Iris dataset to S3
WORK_DIRECTORY = "data"
train_input = sagemaker_session.upload_data(
    WORK_DIRECTORY, key_prefix="{}/{}".format(prefix, WORK_DIRECTORY)
)

# Define the SKLearn estimator
FRAMEWORK_VERSION = "1.0-1"
script_path = "sklearn.py"
sklearn = SKLearn(
    entry_point=script_path,
    framework_version=FRAMEWORK_VERSION,
    instance_type="ml.m5.large",
    role=role,
    sagemaker_session=sagemaker_session,
    hyperparameters={"max_leaf_nodes": 30},
)

# Train the SKLearn estimator
sklearn.fit({"train": train_input})

# Deploy the SKLearn estimator
predictor = sklearn.deploy(initial_instance_count=1, instance_type="ml.m5.large")

# Test the SKLearn estimator
shape = pd.read_csv("data/iris.csv", header=None)
a = [50 * i for i in range(3)]
b = [40 + i for i in range(10)]
indices = [i + j for i, j in itertools.product(a, b)]
test_data = shape.iloc[indices[:-1]]
test_X = test_data.iloc[:, 1:]
test_y = test_data.iloc[:, 0]

# Predict the values
print(predictor.predict(test_X.values))
print(test_y.values)

# Delete the endpoint
predictor.delete_endpoint()