"""
Synthetic sales series generator
"""

import numpy as np
import pandas as pd
from src.data_layer.schema import COVARIATE_PREFIX

def generate_series(
    n_days: int,
    start_date: str = "2022-01-01",
    base_level: float = 100.0,
    trend_per_day: float = 0.05,
    weekly_amplitude: float = 15.0,
    noise_std: float = 5.0,
    promo_prob: float = 0.03,
    promo_multiplier: float = 1.8,
    include_covariates: bool = False,
    promo_effect_size: float = 50.0,
    seed: int | None = None,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_idx = np.arange(n_days)

    trend = trend_per_day * day_idx

    day_of_week = dates.dayofweek.values
    weekly_pattern = weekly_amplitude * np.sin(2 * np.pi * (day_of_week / 7.0))

    signal = base_level + trend + weekly_pattern
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_days)
    values = signal + noise

    # Part 1: Handling spikes in the target variable -- Simulates real world randomness

    spike = rng.random(n_days) < promo_prob
    values = np.where(spike, values * promo_multiplier, values)

    # Part 2: Handle known covariate effects -- Simulates explainable data spikes
    covariate_promo = None
    if include_covariates:
        covariate_promo = (rng.random(n_days) < promo_prob).astype(float)
        values = values + covariate_promo * promo_effect_size

    values = np.clip(values, a_min=0, a_max=None)

    df = pd.DataFrame({"timestamp": dates, "value": values})
    if include_covariates:
        df["covariate_promo"] = covariate_promo

    return df


def generate_dataset(
    n_series: int,
    n_days: int,
    seed: int | None = None,
    **series_kwargs,
) -> pd.DataFrame:
    """
    Generate a multi-series synthetic dataset.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for i in range(n_series):
        series_seed = int(rng.integers(0, 1_000_000))
        df = generate_series(n_days=n_days, seed=series_seed, **series_kwargs)
        df["series_id"] = f"series_{i:03d}"
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    base_cols = ["timestamp", "series_id", "value"]
    covariate_cols = [c for c in out.columns if c.startswith(COVARIATE_PREFIX)]
    return out[base_cols + covariate_cols]


if __name__ == "__main__":
    # univariate
    df1 = generate_dataset(n_series=2, n_days=30, seed=0)
    print("Phase 1 (no covariates):")
    print(df1.head())
    print("columns:", list(df1.columns))
    print()

    # covariates with known effects
    df2 = generate_dataset(
        n_series=2, n_days=30, seed=0,
        include_covariates=True, promo_effect_size=50.0,
    )
    print("Phase 2 (with covariates):")
    print(df2.head())
    print("columns:", list(df2.columns))
    print()