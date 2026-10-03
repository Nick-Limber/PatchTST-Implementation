import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

PRICE_COLS = ["adj_close", "adj_high", "adj_open", "adj_low"]

DROP_COLS = [
    "open", "high", "low", "close",
    "split_factor", "dividend",
    "exchange", "name", "asset_type",
    "price_currency", "exchange_code",
    "adj_volume",
]

def read_s3(
    access_key: str,
    secret_key: str,
    bucket_name: str,
    region: str,
    read_path: str,
) -> pd.DataFrame:
    storage_options = {
        "key":    access_key,
        "secret": secret_key,
        "client_kwargs": {"region_name": region},
    }
    df = pd.read_parquet(
        path=f"s3://{bucket_name}/{read_path}",
        storage_options=storage_options,
    )
    print(f"Read {len(df)} rows from s3://{bucket_name}/{read_path}")
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Before cleaning: {df.shape}")

    for adj_col, raw_col in [
        ("adj_close", "close"),
        ("adj_high",  "high"),
        ("adj_open",  "open"),
        ("adj_low",   "low"),
    ]:
        df[adj_col] = df[adj_col].fillna(
            (df[raw_col] - df["dividend"]) / df["split_factor"]
        )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize().dt.tz_localize(None)

    null_counts          = df[PRICE_COLS].isnull().sum()
    zero_negative_counts = (df[PRICE_COLS] <= 0).sum()
    issues               = null_counts + zero_negative_counts

    if (issues > 0).any():
        print(
            f"Issues detected:\n"
            f"Nulls:\n{null_counts}\n"
            f"Zeros/negatives:\n{zero_negative_counts}"
        )

    df[PRICE_COLS] = df[PRICE_COLS].mask(df[PRICE_COLS] <= 0, np.nan)

    df = df.sort_values(["symbol", "date"])

    df[PRICE_COLS] = (
        df.groupby("symbol")[PRICE_COLS]
        .transform(lambda x: x.ffill(limit=3))
    )

    remaining = df[PRICE_COLS].isnull().sum()
    if (remaining > 0).any():
        print(f"Unfillable gaps remaining after ffill:\n{remaining[remaining > 0]}")
        df = df.dropna(subset=["adj_close"])

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df["log_return"] = np.log(
        df["adj_close"] / df.groupby("symbol")["adj_close"].shift(1)
    )

    df["covariate_daily_range"] = (
        (df["adj_high"] - df["adj_low"]) / df["adj_close"]
    )

    df = df.dropna(subset=["log_return"])

    df = df.rename(columns={
        "date":      "timestamp",
        "symbol":    "series_id",
        "log_return": "value",
        "volume":    "covariate_volume",
        "adj_close": "covariate_adj_close",
        "adj_high":  "covariate_adj_high",
        "adj_low":   "covariate_adj_low",
        "adj_open":  "covariate_adj_open",
    })

    base_cols      = ["timestamp", "series_id", "value"]
    covariate_cols = sorted([c for c in df.columns if c.startswith("covariate_")])
    df = df[base_cols + covariate_cols]

    print(f"After cleaning: {df.shape}")
    return df


def write_data_s3(
    df: pd.DataFrame,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    region: str,
    write_path: str,
    partition_date: str,
) -> str:
    storage_options = {
        "key":    access_key,
        "secret": secret_key,
        "client_kwargs": {"region_name": region},
    }
    path = f"s3://{bucket_name}/{write_path}/{partition_date}.parquet"
    df.to_parquet(
        path=path,
        engine="pyarrow",
        compression="snappy",
        storage_options=storage_options,
        index=False,
    )
    print(f"Written {len(df)} rows to {path}")
    return path



def run_stock_transform(**context) -> str:
    aws_access     = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret     = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region     = os.environ["AWS_REGION"]
    bucket         = os.environ["S3_BUCKET_NAME"]
    partition_date = context.get("ds", date.today().isoformat())

    df = read_s3(
        aws_access, aws_secret, bucket, aws_region,
        f"raw/{partition_date}.parquet"
    )
    df = clean_df(df)
    path = write_data_s3(
        df=df,
        access_key=aws_access,
        secret_key=aws_secret,
        bucket_name=bucket,
        region=aws_region,
        write_path="processed/stock",
        partition_date=partition_date,
    )
    return path
