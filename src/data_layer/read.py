import pandas as pd

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

def get_stats(df):
    
    print(f" datatypes: {df.info()}")
    print(f" total rows: {len(df)}")

    symbols = df["symbol"].unique()
    print(f"total symbols: {symbols}")

    print(f"start date: {df['date'].min()}")
    print(f"end date: {df['date'].max()}")

    print(f"ROWS PER TICKER SYMBOL")
    print(df.groupby("symbol").size())

    print(df[df["asset_type"].notna()].head())
    # ADD MORE EXPLORATION AS NEEDED


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
    get_stats(df)

