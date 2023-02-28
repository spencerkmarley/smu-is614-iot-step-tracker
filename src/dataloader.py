import boto3
import pandas as pd
import time
import numpy as np
import awswrangler
from config import MLCONFIG, QUERY


class DataLoader:
    def __init__(
        self
    ) -> None:

        self.data = pd.DataFrame()
        self.client = boto3.client("athena")

    def load_data(self, query, database) -> None:
        """
        Load data from AWS Athena

            Args:
                query: query for data extraction from athena
                database: name of database for data extraction
            Returns:
                None
        """
        try:
            self.data = awswrangler.athena.read_sql_query(sql=query, database=database)
            print ("Query is successful!")
        except:
            print (f"Unable to connect to {database}")

        return self.data

    def display_data(self) -> None:
        print (self.data)

    def get_data(self) -> pd.DataFrame:
        return self.data


if __name__ == "__main__":
    dataloader = DataLoader()
    dataloader.load_data(QUERY.RAW_DATA, 'smu-iot')
    dataloader.display_data()
