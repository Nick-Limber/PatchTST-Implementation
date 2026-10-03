import os
import requests
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

TICKER_SYMBOLS = [
    "XLK", "XLC", "XLE",
    "AAPL", "MSFT",
    "GOOGL", "META",
    "XOM", "SPY",
]


def ingest_eod(
    api_key: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch EOD data from Marketstack with automatic pagination."""
    total_data = []
    offset     = 0
    url        = "https://api.marketstack.com/v2/eod"
    params     = {
        "access_key": api_key,
        "symbols":    ",".join(symbols) if isinstance(symbols, list) else symbols,
        "limit":      limit,
        "date_from":  start_date,
        "date_to":    end_date,
    }

    while True:
        params["offset"] = offset
        response = requests.get(url, params=params)
        response.raise_for_status()
        payload    = response.json()
        pagination = payload.get("pagination", {})
        data       = payload.get("data", [])

        if not data:
            break

        total_data.extend(data)

        count = pagination.get("count", 0)
        total = pagination.get("total", 0)

        if offset + count >= total:
            break

        offset += count

    return pd.DataFrame(total_data)


def write_raw_s3(
    df: pd.DataFrame,
    bucket_name: str,
    access_key: str,
    secret_key: str,
    region: str,
    write_path: str,
    partition_date: str | None = None,
) -> str:
    partition_date  = partition_date or date.today().isoformat()
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


def run_stock_ingestion(**context) -> str:

    api_key     = os.environ["MARKETSTACK_API_KEY"]
    bucket      = os.environ["S3_BUCKET_NAME"]
    aws_access  = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret  = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region  = os.environ["AWS_REGION"]

    run_date    = context.get("ds", (date.today() - timedelta(days=1)).isoformat())
    yesterday   = (date.fromisoformat(run_date) - timedelta(days=1)).isoformat()

    df = ingest_eod(
        api_key=api_key,
        symbols=TICKER_SYMBOLS,
        start_date=yesterday,
        end_date=run_date,
    )

    path = write_raw_s3(
        df=df,
        bucket_name=bucket,
        access_key=aws_access,
        secret_key=aws_secret,
        region=aws_region,
        partition_date=run_date,
    )

    return path
