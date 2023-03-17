from typing import Dict, Tuple
import numpy.typing as npt
import numpy as np
import awswrangler as wr
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseShuffleSplit
from datetime import datetime
from src.config import PATHS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import argparse
import boto3
import joblib
from io import BytesIO
from src.config import MLCONFIG, KEYS
from src.model import BaseModel
from src.dataloader import DataLoader
from src.feature_generator import FeatureEngineering

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", type=str, default="testing", required=False, choices=["lr", "rf"]
    )
    parser.add_argument(
        "--model", type=str, default="mlflow-artifact-store/"
                                     "179367357541384069/"
                                     "6e127e9d1012490e9bebc3e4695a4bbd/"
                                     "artifacts/model/model.pkl",
        required=False
    )
    parser.add_argument(
        "--s3bucket", type=str, default="smu-is614-iot-step-tracker",
        required=False
    )
    parser.add_argument(
        "--result", type=str, default="inference/result",
        required=False
    )

    args = parser.parse_args()

    # load data via athena
    session = boto3.setup_default_session(
        region_name=KEYS.AWS_DEFAULT_REGION,
        aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
    )
    # setup s3
    s3 = boto3.client(
        's3',
        region_name=KEYS.AWS_DEFAULT_REGION,
        aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
    )

    dataloader = DataLoader(session=session)
    QUERY = f"""
        SELECT
            *
        FROM
            "smu-iot"."microbit"
        WHERE
            seconds IS NOT null
            AND
            uuid LIKE '%{args.query}%'
        ORDER BY
            uuid, timestamp, seconds
    """

    df = dataloader.load_data(QUERY, "smu-iot")

    fe_settings = {
        "upload_to_s3": True,
        "apply_smooth_filter": True,
        "apply_median_filter": True,
        "apply_savgol_filter": True,
        "extract_features": True,
        "window_duration": 4,
        "step_seconds": 0.07,
    }
    FE = FeatureEngineering(**fe_settings)
    X, X_full = FE.fit_transform(df, df.uuid)

    try:
        with BytesIO() as f:
            s3.download_fileobj(Bucket=args.s3bucket, Key=args.model, Fileobj=f)
            f.seek(0)
            clf = joblib.load(f)
        print("Model loading successful.")
    except:
        print ("Model loading error!")

    try:
        X_full['target_label'] = clf.predict(X).astype(int)
        X_full['target'] = (X_full['target_label'] == 1)
        print ("Inference successful.")
    except:
        print ("Inference error!")

    try:
        wr.s3.to_csv(
            df=X_full,
            path=f"s3://{args.s3bucket}/{args.result}/{X_full['timestamp'].max()}.csv",
            index=False,
        )
        print ("Uploaded inference result to s3.")
    except:
        print ("Unable to upload inference result to s3")
=======
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
