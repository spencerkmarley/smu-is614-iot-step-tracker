import time
import boto3
from config import QUERY, KEYS

RESULT_OUTPUT_LOCATION = "s3://smu-is614-iot-step-tracker/queries/"


def setup_default_clients():
    athena = boto3.client(
        "athena",
        region_name=KEYS.AWS_DEFAULT_REGION,
        aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
    )
    s3 = boto3.client(
        "s3",
        region_name=KEYS.AWS_DEFAULT_REGION,
        aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
    )
    return s3, athena


def check_query_status(client, execution_id):
    state = "RUNNING"
    max_execution = 5

    while max_execution > 0 and state in ["RUNNING", "QUEUED"]:
        max_execution -= 1
        response = client.get_query_execution(QueryExecutionId=execution_id)
        if (
            "QueryExecution" in response
            and "Status" in response["QueryExecution"]
            and "State" in response["QueryExecution"]["Status"]
        ):
            state = response["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                return True

        time.sleep(5)

    return False


def generate_response(client, query: str, output: str) -> None:
    response = client.start_query_execution(
        QueryString=query, ResultConfiguration={"OutputLocation": output}
    )

    time.sleep(0.5)

    if check_query_status(client, response["QueryExecutionId"]):
        result = client.get_query_results(QueryExecutionId=response["QueryExecutionId"])
        print("query is successful!")
        result = result["ResultSet"]["Rows"]

    print("No result returned!")
    return None


if __name__ == "__main__":
    s3, athena = setup_default_clients()
    generate_response(
        client=athena, query=QUERY.RAW_DATA, output=RESULT_OUTPUT_LOCATION
    )
