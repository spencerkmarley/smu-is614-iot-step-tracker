import boto3
import pandas as pd
import time
import numpy as np
from config import MLCONFIG, QUERY


class DataLoader:
    def __init__(
        self,
        bucket_name: str,
        data_key: str,
        test_size: float = 0.2,
        random_state: str = MLCONFIG.RANDOM_STATE,
    ) -> None:
        self.bucket_name = bucket_name
        self.data_key = data_key
        self.test_size = test_size
        self.random_state = random_state
        self.client = boto3.client("athena")

    def check_query_status(self, execution_id):
        state = "RUNNING"
        max_execution = 5

        while max_execution > 0 and state in ["RUNNING", "QUEUED"]:
            max_execution -= 1
            response = self.client.get_query_execution(
                QueryExecutionId=execution_id
            )
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

    def generate_response(self, query: str = QUERY.RAW_DATA, output: str) -> None:
        response = self.start_query_execution(
            QueryString=query, ResultConfiguration={"OutputLocation": output}
        )

        time.sleep(1)

        if self.check_query_status(self.client, response["QueryExecutionId"]):
            result = self.client.get_query_results(
                QueryExecutionId=response["QueryExecutionId"]
            )
            print("query is successful!")
            self.result = result["ResultSet"]["Rows"]

        print("No result returned!")
        return None

    def load_data(self) -> None:
        cols = {}
        for c in self.result[0]["Data"]:
            k = c["VarCharValue"]
            cols[k] = None

        self.data = np.array(
            [
                [result["VarCharValue"] for result in self.result[i]["Data"]]
                for i in range(1, len(self.result))
            ]
        )


if __name__ == "__main__":
    dataloader = DataLoader()
