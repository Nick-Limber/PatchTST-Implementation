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

    print(f" datatypes: {df.info()}")
    print(f" total rows: {len(df)}")

    df["adj_close"] = df["adj_close"].fillna((df["close"] - df["dividend"]) / df["split_factor"])
    df["adj_high"] = df["adj_high"].fillna((df["high"] - df["dividend"]) / df["split_factor"])
    df["adj_open"] = df["adj_open"].fillna((df["open"] - df["dividend"]) / df["split_factor"])
    df["adj_low"] = df["adj_low"].fillna((df["low"] - df["dividend"]) / df["split_factor"])

    zero_check_cols = ["adj_close", "adj_high", "adj_open", "adj_low"]
    
    # Should be no null values after the above transformations
    null_counts = df[zero_check_cols].isnull.sum()
    zero_negative_count = (df[zero_check_cols] <= 0).sum()
    issues = null_counts + zero_negative_count

    if (issues > 0):
        print(f"nulls: {null_counts}, zeros and negatives: {zero_negative_count}")



    drop_cols = [ "open", "high", "low", "close", "split_factor", "dividend", "exchange", "name", "asset_type", "price_currency", "exchange_code" ]
    df = df.drop(columns=drop_cols)

    print(f" datatypes: {df.info()}")
    print(f" total rows: {len(df)}")
    
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))


def load_data_s3(df, access_key, secret_key, bucket_name, region, write_path):
        
    
    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {
            "region_name": region
        }
    }

    df.to_parquet(
            path=f"s3://{bucket_name}/{read_path}.parquet",
            engine="pyarrow",
            compression="snappy",
            storage_options = storage_options
            )


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv

    load_dotenv()
    
    aws_access = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region = os.environ["AWS_REGION"]
    bucket     = os.environ["S3_BUCKET_NAME"]
    read_path = "raw"
    write_path = "processed"
    
    df = read_s3(aws_access, aws_secret, bucket, aws_region, read_path)
    clean_df(df)
    load_data_s3(df, aws_access, aws_secret, bucket, aws_region, write_path)
