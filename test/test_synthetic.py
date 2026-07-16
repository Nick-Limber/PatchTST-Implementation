"""
Tests for src/data_layer/synthetic.py

These tests exist to catch bugs in the data generator itself, before
that data ever reaches a training loop.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_layer.synthetic import generate_series, generate_dataset
from src.data_layer.schema import REQUIRED_COLUMNS, COVARIATE_PREFIX


# Phase 1 (no covariates)

def test_phase1_has_only_required_columns():

    df = generate_dataset(n_series=2, n_days=10, seed=0)
    assert list(df.columns) == REQUIRED_COLUMNS

def test_phase1_row_count_matches_series_and_days():
    
    n_series, n_days = 3, 14
    df = generate_dataset(n_series=n_series, n_days=n_days, seed=0)
    assert len(df) == n_series * n_days
    counts = df.groupby("series_id").size()
    assert (counts == n_days).all()

def test_values_are_never_negative():

    df = generate_dataset(
        n_series=5, n_days=200, seed=1,
        noise_std=50.0, trend_per_day=-2.0,
    )
    assert (df["value"] >= 0).all()

def test_series_ids_are_unique_and_correctly_formatted():
    
    df = generate_dataset(n_series=12, n_days=5, seed=0)
    unique_ids = sorted(df["series_id"].unique())
    expected_ids = [f"series_{i:03d}" for i in range(12)]
    assert unique_ids == expected_ids

# Reproducibility

def test_same_seed_produces_identical_output():
    
    df_a = generate_dataset(n_series=3, n_days=20, seed=42)
    df_b = generate_dataset(n_series=3, n_days=20, seed=42)
    pd.testing.assert_frame_equal(df_a, df_b)

def test_different_seed_produces_different_output():
    
    df_a = generate_dataset(n_series=3, n_days=20, seed=42)
    df_b = generate_dataset(n_series=3, n_days=20, seed=43)

    assert not df_a["value"].equals(df_b["value"])

# Phase 2 (covariates)

def test_phase2_adds_covariate_column():
    
    df = generate_dataset(
        n_series=2, n_days=10, seed=0, include_covariates=True,
    )
    assert "covariate_promo" in df.columns
    assert df["covariate_promo"].isin([0.0, 1.0]).all()

def test_phase2_covariate_columns_discovered_via_prefix():

    df = generate_dataset(
        n_series=2, n_days=10, seed=0, include_covariates=True,
    )
    discovered = [c for c in df.columns if c.startswith(COVARIATE_PREFIX)]
    assert discovered == ["covariate_promo"]

def test_phase2_known_effect_is_recoverable():

    effect_size = 50.0
    df = generate_dataset(
        n_series=10, n_days=300, seed=0,
        include_covariates=True, promo_effect_size=effect_size,
    )
    grouped = df.groupby("covariate_promo")["value"].mean()
    observed_diff = grouped[1.0] - grouped[0.0]

    # Allow a reasonably wide tolerance -- this is a statistical estimate
    assert observed_diff == pytest.approx(effect_size, rel=0.3)

def test_phase1_and_phase2_share_identical_base_signal_under_same_seed():

    df1 = generate_dataset(n_series=1, n_days=30, seed=7)
    df2 = generate_dataset(
        n_series=1, n_days=30, seed=7, include_covariates=True,
        promo_effect_size=0.0,
    )
    pd.testing.assert_series_equal(
        df1["value"], df2["value"], check_names=False
    )

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))