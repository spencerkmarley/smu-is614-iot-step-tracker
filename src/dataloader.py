import boto3
import pandas as pd
import awswrangler as wr
from awswrangler.exceptions import QueryFailed
from src.config import KEYS, PATHS, QUERY


class DataLoader:
    def __init__(self, session: boto3.Session) -> None:
        self.session = session

    def load_data(self, query: str, database: str) -> None:
        """
        Load data from AWS Athena
        Args:
            query: query for data extraction from athena
            database: name of database for data extraction
        Returns:
            None
        """
        try:
            self.data = wr.athena.read_sql_query(
                sql=query, database=database, boto3_session=self.session
            )
        except QueryFailed as err:
            raise QueryFailed(err)

        return self.data

    def get_data(self) -> pd.DataFrame:
        try:
            return self.data
        except AttributeError as err:
            raise AttributeError(err)


if __name__ == "__main__":
    session = boto3.setup_default_session(
        region_name=KEYS.AWS_DEFAULT_REGION,
        aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
    )
    dataloader = DataLoader(session=session)
    dataloader.load_data(query=QUERY.RAW_DATA, database=PATHS.DB_TEST)
    print(dataloader.get_data())
