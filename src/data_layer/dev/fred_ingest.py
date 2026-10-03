import pandas as pd
from fredapi import Fred
import os
import time
from datetime import date

FRED_SERIES = {
    "covariate_vix":          {"id": "VIXCLS",   "lag_days": 0},
    "covariate_treasury_10y": {"id": "DGS10",    "lag_days": 0},
    "covariate_treasury_2y":  {"id": "DGS2",     "lag_days": 0},
    "covariate_fed_funds":    {"id": "FEDFUNDS", "lag_days": 35},
    "covariate_cpi":          {"id": "CPIAUCSL", "lag_days": 14},
    "covariate_pce":          {"id": "PCEPI",    "lag_days": 28},
    "covariate_unemployment": {"id": "UNRATE",   "lag_days": 7},
}


def ingest_transform_fred(start_date: str = "2020-01-01") -> pd.DataFrame:
    api_key = os.environ["FRED_API_KEY"]
    fred    = Fred(api_key=api_key)

    frames = []
    for col_name, config in FRED_SERIES.items():
        print(f"Fetching {config['id']} → {col_name}...")
        series = fred.get_series(
            config["id"],
            observation_start=start_date,
        )

        if config["lag_days"] > 0:
            series.index = series.index + pd.Timedelta(days=config["lag_days"])

        frames.append(series.rename(col_name).to_frame())
        time.sleep(0.5)

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    if "covariate_treasury_10y" in combined.columns and \
       "covariate_treasury_2y" in combined.columns:
        combined["covariate_yield_spread"] = (
            combined["covariate_treasury_10y"] -
            combined["covariate_treasury_2y"]
        )

    combined = combined.ffill()
    return combined


def read_s3(
    access_key: str,
    secret_key: str,
    bucket_name: str,
    region: str,
    read_path: str,
) -> pd.DataFrame:
    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {"region_name": region},
    }
    return pd.read_parquet(
        path=f"s3://{bucket_name}/{read_path}",
        storage_options=storage_options,
    )


def merge_datasets(
    stock_df: pd.DataFrame,
    fred_df: pd.DataFrame,
) -> pd.DataFrame:
    if stock_df.empty:
        raise ValueError("Stock DataFrame is empty.")

    stock_indexed = stock_df.set_index("timestamp")
    stock_indexed.index = pd.to_datetime(stock_indexed.index)
    if stock_indexed.index.tz is not None:
        stock_indexed.index = stock_indexed.index.tz_localize(None)

    fred_df = fred_df.copy()
    fred_df.index = pd.to_datetime(fred_df.index)
    if fred_df.index.tz is not None:
        fred_df.index = fred_df.index.tz_localize(None)

    merged = stock_indexed.join(fred_df, how="left")

    covariate_cols = [c for c in merged.columns if c.startswith("covariate_")]
    merged[covariate_cols] = merged[covariate_cols].ffill()

    remaining_nulls = merged[covariate_cols].isnull().sum()
    if (remaining_nulls > 0).any():
        print(f"Warning: remaining nulls after ffill:\n{remaining_nulls[remaining_nulls > 0]}")

    print(f" datatypes: {merged.info()}")
    print(f" total rows: {len(merged)}")

    return merged.reset_index()


def write_data_s3(
    df: pd.DataFrame,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    region: str,
    write_path: str,
) -> None:
    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {"region_name": region},
    }
    path=f"s3://{bucket_name}/{write_path}/{date.today().isoformat()}.parquet"
    df.to_parquet(
        path=path,
        engine="pyarrow",
        compression="snappy",
        storage_options=storage_options,
        index=False,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    aws_access = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region = os.environ["AWS_REGION"]
    bucket     = os.environ["S3_BUCKET_NAME"]
    write_path = "processed/combined"

    fred_df = ingest_transform_fred(start_date="2020-01-01")
    print(f"FRED data: {fred_df.shape}, date range: {fred_df.index.min()} to {fred_df.index.max()}")

    stock_df = read_s3(aws_access, aws_secret, bucket, aws_region, "processed/canonical")
    print(f"Stock data: {stock_df.shape}")

    merged_df = merge_datasets(stock_df, fred_df)
    print(f"Merged: {merged_df.shape}")
    print(merged_df.head())

    write_data_s3(merged_df, aws_access, aws_secret, bucket, aws_region, write_path)


