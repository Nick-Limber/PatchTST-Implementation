import os

from dotenv import load_dotenv

from theta_client import ingest_day

load_dotenv()


def run(iso_date: str) -> str | None:
    bucket = os.environ["S3_BUCKET_NAME"]
    return ingest_day(bucket, iso_date)


if __name__ == "__main__":
    # Manual/ad hoc run only -- the DAG passes iso_date itself.
    from datetime import date
    run(date.today().strftime("%Y-%m-%d"))
