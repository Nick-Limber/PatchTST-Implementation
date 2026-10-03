from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


IV_RANK_WINDOW_DAYS = 252        # trailing window for IV rank/percentile
REALIZED_VOL_WINDOWS = [5, 20]   # trailing windows for underlying realized vol
MOMENTUM_WINDOWS = [1, 5, 20]    # trailing windows for underlying returns
ATM_MONEYNESS_BAND = 0.03        # +/-3% of spot counts as "ATM" for the IV-history series
LABEL_HORIZON_DAYS = 5           # holding period for the profitability label
ASSUMED_ROUND_TRIP_COST = 0.01   # 1% of premium, covers spread/slippage -- tune to your instrument


@dataclass
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(f"s3://{bucket}/{key}")


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> str:
    df.to_parquet(f"s3://{bucket}/{key}", index=False)
    return f"s3://{bucket}/{key}"


def read_bucket_prefix(bucket: str, prefix: str, filters: list | None = None) -> pd.DataFrame:
    dataset = ds.dataset(f"s3://{bucket}/{prefix}", format="parquet", partitioning="hive")
    return dataset.to_table(filter=_pyarrow_filter(filters) if filters else None).to_pandas()


def _pyarrow_filter(filters: list):
    import pyarrow.compute as pc
    expr = None
    for col, op, val in filters:
        cond = {"=": pc.equal, "!=": pc.not_equal, ">": pc.greater,
                ">=": pc.greater_equal, "<": pc.less, "<=": pc.less_equal}[op](ds.field(col), val)
        expr = cond if expr is None else expr & cond
    return expr


def _raw_chain_path(bucket: str, iso_date: str) -> tuple[str, str]:
    year, month, _ = iso_date.split("-")
    return bucket, f"historical/options/year={year}/month={month}/{iso_date}.parquet"


def clean_daily_chain(bucket: str, iso_date: str) -> str:
    raw_bucket, raw_key = _raw_chain_path(bucket, iso_date)
    df = _read_parquet_s3(raw_bucket, raw_key)

    if df.empty:
        print(f"{iso_date}: raw chain was empty, writing an empty cleaned file")
        out_key = f"cleaned/options/date={iso_date}.parquet"
        return _write_parquet_s3(df, bucket, out_key)

    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    for col in ["strike", "close", "open", "high", "low", "underlying_close",
                "delta", "gamma", "theta", "vega", "rho", "implied_volatility", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) > 0]
    df = df.dropna(subset=["close", "underlying_close", "strike"])

    if "implied_volatility" in df.columns:
        missing_greeks_frac = df["implied_volatility"].isna().mean()
        if missing_greeks_frac > 0.15:
            print(f"WARNING: {iso_date} has {missing_greeks_frac:.1%} rows missing IV/Greeks -- "
                  f"investigate before trusting this day's features")

    df = df.sort_values(["underlying_ticker", "expiration", "strike"])

    after = len(df)
    print(f"{iso_date}: cleaned {before} -> {after} rows ({before - after} dropped)")

    out_key = f"cleaned/options/date={iso_date}.parquet"
    return _write_parquet_s3(df, bucket, out_key)



def _load_trailing_underlying_history(bucket: str, tickers: list[str], end_date: str,
                                       lookback_days: int) -> pd.DataFrame:
    dates = pd.bdate_range(end=end_date, periods=lookback_days + 1)
    rows = []
    for d in dates:
        iso = d.strftime("%Y-%m-%d")
        try:
            day_df = _read_parquet_s3(bucket, f"cleaned/options/date={iso}.parquet")
        except Exception:
            continue  # missing day (holiday, gap in backfill) -- just skip it
        day_df = day_df[day_df["underlying_ticker"].isin(tickers)]

        for ticker, g in day_df.groupby("underlying_ticker"):
            underlying_close = g["underlying_close"].iloc[0]

            g = g.assign(moneyness=(g["strike"] / underlying_close - 1).abs())
            atm = g[g["moneyness"] <= ATM_MONEYNESS_BAND].sort_values(
                ["expiration", "moneyness"])
            atm_iv = atm["implied_volatility"].iloc[0] if not atm.empty else np.nan

            rows.append({"date": iso, "underlying_ticker": ticker,
                         "underlying_close": underlying_close, "atm_iv": atm_iv})

    return pd.DataFrame(rows)


def _underlying_rolling_features(history: pd.DataFrame) -> pd.DataFrame:
    history = history.sort_values(["underlying_ticker", "date"]).copy()
    out = []
    for ticker, g in history.groupby("underlying_ticker"):
        g = g.copy()
        g["log_return"] = np.log(g["underlying_close"]).diff()

        for w in REALIZED_VOL_WINDOWS:
            g[f"realized_vol_{w}d"] = g["log_return"].rolling(w).std() * np.sqrt(252)
        for w in MOMENTUM_WINDOWS:
            g[f"momentum_{w}d"] = g["underlying_close"].pct_change(w)

        g["iv_rank"] = g["atm_iv"].rolling(IV_RANK_WINDOW_DAYS, min_periods=20).apply(
            lambda s: (s.iloc[-1] - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.nan
        )
        g["iv_percentile"] = g["atm_iv"].rolling(IV_RANK_WINDOW_DAYS, min_periods=20).apply(
            lambda s: (s < s.iloc[-1]).mean()
        )
        if "realized_vol_20d" in g.columns:
            g["vol_risk_premium"] = g["atm_iv"] - g["realized_vol_20d"]

        out.append(g)
    return pd.concat(out, ignore_index=True)


def engineer_features(bucket: str, iso_date: str, clean_path: str) -> str:
    df = _read_parquet_s3(bucket, f"cleaned/options/date={iso_date}.parquet")
    if df.empty:
        out_key = f"features/options/date={iso_date}.parquet"
        return _write_parquet_s3(df, bucket, out_key)

    tickers = df["underlying_ticker"].unique().tolist()

    history = _load_trailing_underlying_history(
        bucket, tickers, iso_date, max(IV_RANK_WINDOW_DAYS, max(REALIZED_VOL_WINDOWS + MOMENTUM_WINDOWS)))
    rolling = _underlying_rolling_features(history)
    today_rolling = rolling[rolling["date"] == iso_date].drop(columns=["date", "underlying_close", "atm_iv"])

    df = df.merge(today_rolling, on="underlying_ticker", how="left")

    df["moneyness"] = np.log(df["strike"] / df["underlying_close"])
    df["days_to_expiry"] = (df["expiration"] - df["date"]).dt.days
    df["dte_bucket"] = pd.cut(df["days_to_expiry"], bins=[-1, 7, 30, 90, 10_000],
                               labels=["weekly", "monthly", "quarterly", "long_dated"])

    near_money = df[df["moneyness"].abs() <= ATM_MONEYNESS_BAND]
    term_structure = (
        near_money.groupby(["underlying_ticker", "expiration"])["implied_volatility"]
        .mean().reset_index().rename(columns={"implied_volatility": "atm_iv_this_expiry"})
        .sort_values(["underlying_ticker", "expiration"])
    )
    term_structure["term_structure_slope"] = (
        term_structure.groupby("underlying_ticker")["atm_iv_this_expiry"].diff()
    )
    df = df.merge(term_structure[["underlying_ticker", "expiration", "term_structure_slope"]],
                   on=["underlying_ticker", "expiration"], how="left")

    df = df.merge(
        near_money.groupby(["underlying_ticker", "expiration"])["implied_volatility"]
        .mean().reset_index().rename(columns={"implied_volatility": "_atm_iv_ref"}),
        on=["underlying_ticker", "expiration"], how="left",
    )
    df["skew"] = df["implied_volatility"] - df["_atm_iv_ref"]
    df = df.drop(columns=["_atm_iv_ref"])

    for col in ["volume", "gamma"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    out_key = f"features/options/date={iso_date}.parquet"
    return _write_parquet_s3(df, bucket, out_key)



def construct_labels(bucket: str, iso_date: str, horizon_days: int = LABEL_HORIZON_DAYS) -> str:

    feat_df = _read_parquet_s3(bucket, f"features/options/date={iso_date}.parquet")

    forward_dates = pd.bdate_range(start=iso_date, periods=horizon_days + 1)[1:]
    forward_frames = []
    for d in forward_dates:
        iso = d.strftime("%Y-%m-%d")
        try:
            forward_frames.append(_read_parquet_s3(bucket, f"cleaned/options/date={iso}.parquet"))
        except Exception:
            continue
    if not forward_frames:
        print(f"{iso_date}: no forward data available yet ({horizon_days}d horizon) -- skipping")
        return ""

    forward_df = pd.concat(forward_frames, ignore_index=True)
    last_forward_date = forward_df["date"].max()

    labeled_rows = []
    for _, row in feat_df.iterrows():
        contract_forward = forward_df[
            (forward_df["option_ticker"] == row["option_ticker"])
            & (forward_df["date"] == last_forward_date)
        ]

        if row["expiration"] <= last_forward_date:
            expiry_underlying = forward_df.loc[
                (forward_df["underlying_ticker"] == row["underlying_ticker"])
                & (forward_df["date"] == row["expiration"]),
                "underlying_close"
            ]
            if expiry_underlying.empty:
                continue
            spot_at_expiry = expiry_underlying.iloc[0]
            exit_value = (max(0.0, spot_at_expiry - row["strike"]) if row["contract_type"] == "call"
                          else max(0.0, row["strike"] - spot_at_expiry))
        elif not contract_forward.empty:
            exit_value = contract_forward["close"].iloc[0]
        else:
            continue  # contract not found in forward data (delisted/no trade) -- drop, don't guess

        entry_cost = row["close"] * (1 + ASSUMED_ROUND_TRIP_COST)
        exit_proceeds = exit_value * (1 - ASSUMED_ROUND_TRIP_COST)
        forward_return = (exit_proceeds - entry_cost) / entry_cost if entry_cost > 0 else np.nan

        labeled_rows.append({
            **row.to_dict(),
            "forward_return": forward_return,
            "label_profitable": forward_return > 0 if pd.notna(forward_return) else None,
            "label_horizon_days": horizon_days,
        })

    labeled_df = pd.DataFrame(labeled_rows)
    out_key = f"labeled/options/date={iso_date}_h{horizon_days}.parquet"
    return _write_parquet_s3(labeled_df, bucket, out_key)


