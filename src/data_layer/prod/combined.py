import os
from datetime import date, timedelta
import pandas as pd

def merge_and_save_datasets(target_date: str = None) -> str:
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

    # Read Stock Data
    stock_path = f"s3://{bucket}/processed/stock/{target_date}.parquet"
    stock_df = pd.read_parquet(stock_path, storage_options=storage_options)

    # Read Transformed FRED Data
    fred_path = f"s3://{bucket}/processed/FRED/{target_date}.parquet"
    fred_df = pd.read_parquet(fred_path, storage_options=storage_options)

    if stock_df.empty:
        raise ValueError("Stock DataFrame is empty.")

    stock_indexed = stock_df.set_index("timestamp")
    stock_indexed.index = pd.to_datetime(stock_indexed.index)
    if stock_indexed.index.tz is not None:
        stock_indexed.index = stock_indexed.index.tz_localize(None)

    fred_df.index = pd.to_datetime(fred_df.index)
    if fred_df.index.tz is not None:
        fred_df.index = fred_df.index.tz_localize(None)

    merged = stock_indexed.join(fred_df, how="left")

    covariate_cols = [c for c in merged.columns if c.startswith("covariate_")]
    merged[covariate_cols] = merged[covariate_cols].ffill()

    remaining_nulls = merged[covariate_cols].isnull().sum()
    if (remaining_nulls > 0).any():
        print(f"Warning: remaining nulls after ffill:\n{remaining_nulls[remaining_nulls > 0]}")

    final_df = merged.reset_index()

    write_path = f"s3://{bucket}/fina/{target_date}.parquet"
    final_df.to_parquet(
        path=write_path,
        engine="pyarrow",
        compression="snappy",
        storage_options=storage_options,
        index=False,
    )
    print(f"Saved final merged dataset to {write_path}")


