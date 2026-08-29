import requests
from datetime import date, timedelta
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

TICKER_SYMBOLS = ["XLK", "XLC", "AAPL", "MSFT", "GOOGL", "META", "SPY"]

def ingest_eod (api_key, symbols, limit=1000, start_date=(date.today() - timedelta(days=1)).isoformat(), end_date=date.today().isoformat()):

    total_data = []
    offset = 0
    url = "https://api.marketstack.com/v2/eod"

    symbols_string = ",".join(symbols) if isinstance(symbols, list) else symbols

    params = { "access_key": api_key,
               "symbols": symbols_string,
               "limit": limit,
               "date_from": start_date,
               "date_to": end_date
              }
   
    while True:
        params["offset"] = offset
        response = requests.get(url, params=params)
        response.raise_for_status()
        total_response  = response.json()

        pagination = total_response.get("pagination", {})
        data       = total_response.get("data", [])

        if not data:
            break

        total_data.extend(data)

        count = pagination.get("count", 0)
        total = pagination.get("total", 0)

        if offset + count >= total:
            break

        offset += count   # use actual count returned, not assumed limit

    df = pd.DataFrame(total_data)
    return df

def load_data_s3(df, access_key, secret_key, bucket_name, region):
        
    
    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {
            "region_name": region
        }
    }

    df.to_parquet(
            path=f"s3://{bucket_name}/raw/{date.today().isoformat()}.parquet",
            engine="pyarrow",
            compression="snappy",
            storage_options = storage_options
            )
    


if __name__ == "__main__":
    import os

    api_key    = os.environ["MARKETSTACK_API_KEY"]
    aws_access = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region = os.environ["AWS_REGION"]
    bucket     = os.environ["S3_BUCKET_NAME"]
    
    if not bucket:
        print("NO BUCKET LOADED")

    print(f"Fetching EOD data for: {TICKER_SYMBOLS}")
    df = ingest_eod(
        api_key=api_key,
        symbols=TICKER_SYMBOLS,
        start_date="2022-01-01"
    )

    print(f"Fetched {len(df)} rows across {df['symbol'].nunique()} tickers")
    print(df.head())

    load_data_s3(
      df=df,
      bucket_name=bucket,
      access_key=aws_access,
      secret_key=aws_secret,
      region=aws_region,
  ) 
    
    print("DONE")
