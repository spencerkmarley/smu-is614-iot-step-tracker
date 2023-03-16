import boto3
import pickle
from src.config import KEYS
from src.dataloader import DataLoader
from src.config import MLCONFIG, PATHS
import pandas as pd
import awswrangler as wr


# 1. setup client
session = boto3.setup_default_session(
    region_name=KEYS.AWS_DEFAULT_REGION,
    aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
)
dataloader = DataLoader(session=session)

# 2. query from inference/interim from Athena >> X
# possibly need to change
QUERY = """
    SELECT
        *,
        case when uuid like '%_walk_%' then true else false end as target
    FROM
        "smu-iot"."inference"
    WHERE seconds IS NOT NULL
    ORDER BY
        uuid, timestamp, seconds
"""

df = dataloader.load_data(QUERY, "smu-iot")


# 3. Filter for X cols >> X_filt
X = df[MLCONFIG.TOP_FEATURES]


# 4. load a model artifact from S3
model_artifact = "s3://smu-is614-iot-step-tracker/mlflow-artifact-store/179367357541384069/6e127e9d1012490e9bebc3e4695a4bbd/artifacts/model/model.pkl"
with open(model_artifact, "rb") as file:
    mdl = pickle.load(file)

# 5. Run predictions on X_filt
y_preds = mdl.predict(X)

# 6. Result join back to X
# I forgot if this is correct lol VVVV SONGHAN
final_df = pd.concat([X, y_preds], axis="columns")

# 7. Upload back to S3 >> inference/result
wr.s3.to_csv(
    df=final_df,
    path=f"s3://{PATHS.RESULT}/{final_df['timestamp'].max()}.csv",  # SONGHAN would suggest maybe save the date within the filename instead of max timestamp
    index=False,
)
