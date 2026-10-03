import os

import pandas as pd
from dotenv import load_dotenv

from theta_client import ingest_day

load_dotenv()


def run_historical_backfill(start_date: str, end_date: str):
    bucket = os.environ["S3_BUCKET_NAME"]
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    for single_date in date_range:
        ingest_day(bucket, single_date.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    run_historical_backfill(start_date="2023-01-01", end_date="2026-01-01")
