import pandas as pd
import numpy as np

def read_s3(access_key, secret_key, bucket_name, region, read_path):

    
    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {
            "region_name": region
        }
    }

    df = pd.read_parquet(
            path=f"s3://{bucket_name}/{read_path}",
            storage_options=storage_options
            )

    return df

def clean_df(df):

    df = df["adjusted_close"].fillna((df["close"] - df["dividend"]) / df["split_factor"])

    drop_cols = [ "open", "high", "low", "close", "last", "split_factor", "dividend", "exchange", "name", "asset_type", "price_currency", "exchange_code" ]
    df = df.drop(columns=drop_cols)

    print(f" datatypes: {df.info()}")
    print(f" total rows: {len(df)}")
    
    df["log_return"] = np.log(df["adjusted_close"] / df["adjusted_close"].shift(1))


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv

    load_dotenv()
    
    aws_access = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region = os.environ["AWS_REGION"]
    bucket     = os.environ["S3_BUCKET_NAME"]
    read_path = "raw"
    
    df = read_s3(aws_access, aws_secret, bucket, aws_region, read_path)

    clean_df(df)

