import json

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

BASE_URL = "http://127.0.0.1:25503/v3"
TICKER_SYMBOLS = ["XLK", "XLC", "AAPL", "MSFT", "GOOGL", "META", "SPY"]

_session = requests.Session()
_session.mount("http://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504],
)))

STRIKE_RANGE_N = 15          # n strikes above/below ATM => up to 2n+1 strikes
RATE_TYPE = "sofr"           # let ThetaData pick the day's SOFR rather than a flat guess
REQUEST_TIMEOUT = 30


def _get_v3(path: str, params: dict) -> list[dict]:
    params = {**params, "format": "ndjson"}
    resp = _session.get(f"{BASE_URL}/{path}", params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 472:
        return []
    resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def get_underlying_eod(symbol: str, date_str: str) -> list[dict]:
    return _get_v3("stock/history/eod", {
        "symbol": symbol,
        "start_date": date_str,
        "end_date": date_str,
    })


def get_option_chain_eod(symbol: str, date_str: str) -> list[dict]:
    return _get_v3("option/history/eod", {
        "symbol": symbol,
        "expiration": "*",
        "strike": "*",
        "start_date": date_str,
        "end_date": date_str,
        "strike_range": STRIKE_RANGE_N,
    })


def get_option_chain_greeks_eod(symbol: str, date_str: str) -> list[dict]:
    return _get_v3("option/history/greeks/eod", {
        "symbol": symbol,
        "expiration": "*",
        "strike": "*",
        "start_date": date_str,
        "end_date": date_str,
        "strike_range": STRIKE_RANGE_N,
        "rate_type": RATE_TYPE,
    })


def fetch_historical_option_chain_for_date(symbol: str, date_str: str) -> pd.DataFrame:
    underlying = get_underlying_eod(symbol, date_str)
    underlying_close = underlying[0].get("close") if underlying else None

    bars = get_option_chain_eod(symbol, date_str)
    greeks = get_option_chain_greeks_eod(symbol, date_str)

    if not bars:
        return pd.DataFrame()

    bars_df = pd.DataFrame(bars)
    bars_df["underlying_ticker"] = symbol
    bars_df["underlying_close"] = underlying_close
    bars_df["date"] = date_str

    if greeks:
        greeks_df = pd.DataFrame(greeks)
        join_keys = [k for k in ["expiration", "strike", "right"] if k in bars_df.columns and k in greeks_df.columns]
        if join_keys:
            merged = bars_df.merge(greeks_df, on=join_keys, how="left", suffixes=("", "_greeks"))
        else:
            print(f"  WARNING: couldn't find common join keys for {symbol} {date_str}; "
                  f"bars columns={list(bars_df.columns)}, greeks columns={list(greeks_df.columns)}")
            merged = bars_df
    else:
        merged = bars_df

    return merged


def ingest_day(bucket: str, iso_date: str) -> str | None:
    theta_date = iso_date.replace("-", "")  # ThetaData wants YYYYMMDD
    print(f"Processing options for: {iso_date}")

    daily_frames = []
    for ticker in TICKER_SYMBOLS:
        df_ticker = fetch_historical_option_chain_for_date(ticker, theta_date)
        if not df_ticker.empty:
            daily_frames.append(df_ticker)
            has_iv = df_ticker["implied_volatility"].notna().sum() if "implied_volatility" in df_ticker else "?"
            print(f"  {ticker}: {len(df_ticker)} contract-days (iv present: {has_iv})")

    if not daily_frames:
        print(f"No data for {iso_date}, skipping write")
        return None

    combined_df = pd.concat(daily_frames, ignore_index=True)
    year, month, _ = iso_date.split("-")
    path = f"s3://{bucket}/historical/options/year={year}/month={month}/{iso_date}.parquet"
    combined_df.to_parquet(path, index=False)  # needs s3fs installed
    print(f"Saved {len(combined_df)} rows for {iso_date} -> {path}")
    return path
