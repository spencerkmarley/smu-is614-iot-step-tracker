import boto3
from src.config import KEYS, QUERY, PATHS
from botocore.client import ClientError
from typing import Dict, List, Tuple


def setup_default_clients() -> Tuple[boto3.client]:
    """
    Create default boto3 clients for Athena and S3.
    Returns a tuple of S3 and Athena
    """
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


def generate_result_from_query(
    athena_client: boto3.client,
    query_string: str,
    database: str = "smu-iot",
    s3_output_location: str = f"s3://{PATHS.QUERIES}",
) -> List[Dict]:
    """
    Queries Athena using the provided query string, database, and S3 location.
    Returns the results of the query as a list of dictionaries.
    """

    # Start query execution
    response = athena_client.start_query_execution(
        QueryString=query_string,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": s3_output_location},
    )

    # Get query execution ID
    query_execution_id = response["QueryExecutionId"]

    while True:
        query_execution = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        query_status = query_execution["QueryExecution"]["Status"]["State"]
        if query_status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break

    if query_status == "SUCCEEDED":
        # Get query results
        result_response = athena_client.get_query_results(
            QueryExecutionId=query_execution_id
        )
        result_rows = result_response["ResultSet"]["Rows"]

        results = []
        for row in result_rows[1:]:
            values = [field.get("VarCharValue", "") for field in row["Data"]]
            print(values)
            result = {}
            for i in range(len(result_rows[0]["Data"])):
                result[result_rows[0]["Data"][i]["VarCharValue"]] = values[i]
            results.append(result)

        return results

    else:
        error_message = query_execution["QueryExecution"]["Status"]["StateChangeReason"]
        raise Exception(f"Query failed: {error_message}")


def check_s3_bucket(s3_resource: boto3.resource, bucket_name: str) -> bool:
    """
    Check whether an S3 bucket exists and is accessible.

    Args:
        s3_resource (boto3.resources.factory.s3.ServiceResource):
            The S3 resource to use.
        bucket_name (str): The name of the bucket to check.

    Returns:
        bool: True if the bucket exists and is accessible, False otherwise.
    """
    try:
        s3_resource.meta.client.head_bucket(Bucket=bucket_name)
        print("Bucket Exists")
        return True
    except ClientError as e:
        # Check error code whether bucket is private
        error_code = int(e.response["Error"]["Code"])
        if error_code == 403:
            print("Private Bucket - Forbidden Access")
            return True
        elif error_code == 404:
            print("Bucket Does Not Exist")
            return False


# def check_query_status(client, execution_id):
#     state = "RUNNING"
#     max_execution = 5

#     while max_execution > 0 and state in ["RUNNING", "QUEUED"]:
#         max_execution -= 1
#         response = client.get_query_execution(QueryExecutionId=execution_id)
#         if (
#             "QueryExecution" in response
#             and "Status" in response["QueryExecution"]
#             and "State" in response["QueryExecution"]["Status"]
#         ):
#             state = response["QueryExecution"]["Status"]["State"]
#             if state == "SUCCEEDED":
#                 return True

#         time.sleep(5)

#     return False

# def generate_response(client, query: str, output: str) -> None:
#     response = client.start_query_execution(
#         QueryString=query, ResultConfiguration={"OutputLocation": output}
#     )

#     time.sleep(0.5)

#     if check_query_status(client, response["QueryExecutionId"]):
#         result = client.get_query_results(
#             QueryExecutionId=response["QueryExecutionId"]
#         )
#         print("Query is successful!")
#         result = result["ResultSet"]["Rows"]

#     print("No result returned")
#     return None


if __name__ == "__main__":
    s3, athena = setup_default_clients()
    generate_result_from_query(
        athena_client=athena,
        query=QUERY.RAW_DATA,
        database="smu-iot",
        s3_output_location=f"s3://{PATHS.QUERIES}",
    )

    s3 = boto3.resource(
        "s3",
        region_name=KEYS.AWS_DEFAULT_REGION,
        aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
    )
    check_s3_bucket(s3, PATHS.BUCKET)
