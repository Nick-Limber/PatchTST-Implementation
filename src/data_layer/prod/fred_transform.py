import os
from datetime import date, timedelta
import pandas as pd

def transform_and_save_fred(target_date: str = None) -> str:
    """Reads raw FRED data from S3, applies transformations, and saves to S3."""
    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).isoformat()

    aws_access = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region = os.environ["AWS_REGION"]
    bucket     = os.environ["S3_BUCKET_NAME"]
    
    storage_options = {
        "key": aws_access,
        "secret": aws_secret,
        "client_kwargs": {"region_name": aws_region},
    }

    raw_path = f"s3://{bucket}/raw/fred/{target_date}.parquet"
    print(f"Reading raw FRED data from {raw_path}...")
    transformed = pd.read_parquet(raw_path, storage_options=storage_options)

    transformed.index = pd.to_datetime(transformed.index)
    transformed = transformed.sort_index()

    if "covariate_treasury_10y" in transformed.columns and \
       "covariate_treasury_2y" in transformed.columns:
        transformed["covariate_yield_spread"] = (
            transformed["covariate_treasury_10y"] -
            transformed["covariate_treasury_2y"]
        )

    transformed = transformed.ffill()

    write_path = f"s3://{bucket}/processed/fred_transform/{target_date}.parquet"
    transformed.to_parquet(write_path, storage_options=storage_options, index=True)
    print(f"Saved transformed FRED data to {write_path}")
    return write_path
