import math
import numpy as np
import pandas as pd
import pytest
import torch

from src.data_layer.synthetic import generate_dataset
from src.training.dataset import SlidingWindowDataset

def test_context_and_target_shapes():
    df = generate_dataset(n_series=1, n_days=60, seed=0)
    context_len = 10
    horizon = 5
    dataset = SlidingWindowDataset(df, context_len=context_len, horizon=horizon)

    context, target, stats = dataset[0]

    assert context.shape == (context_len,)
    assert target.shape == (horizon,)
    assert stats.shape == (2,)


def test_output_dtype_is_float32():

    df = generate_dataset(n_series=1, n_days=60, seed=0)
    dataset = SlidingWindowDataset(df, context_len=10, horizon=5)
    context, target, stats = dataset[0]

    assert context.dtype == torch.float32
    assert target.dtype == torch.float32
    assert stats.dtype == torch.float32


def test_window_count_is_correct():
    n_days = 60
    context_len = 10
    horizon = 5
    df = generate_dataset(n_series=1, n_days=n_days, seed=0)
    dataset = SlidingWindowDataset(df, context_len=context_len, horizon=horizon)

    expected = n_days - (context_len + horizon) + 1
    assert len(dataset) == expected


def test_window_count_scales_with_n_series():
    """Total windows = per-series windows * n_series when all series
    have the same length."""
    n_series = 3
    n_days = 60
    context_len = 10
    horizon = 5
    df = generate_dataset(n_series=n_series, n_days=n_days, seed=0)
    dataset = SlidingWindowDataset(df, context_len=context_len, horizon=horizon)

    windows_per_series = n_days - (context_len + horizon) + 1
    assert len(dataset) == n_series * windows_per_series


def test_window_count_with_stride_greater_than_1():

    n_days = 61  
    context_len = 10
    horizon = 5
    stride = 2
    df = generate_dataset(n_series=1, n_days=n_days, seed=0)
    dataset = SlidingWindowDataset(
        df, context_len=context_len, horizon=horizon, stride=stride
    )

    n_valid = n_days - (context_len + horizon) + 1
    expected = math.ceil(n_valid / stride)
    assert len(dataset) == expected


def test_normalized_context_has_mean_zero():
    df = generate_dataset(n_series=2, n_days=60, seed=0)
    dataset = SlidingWindowDataset(df, context_len=10, horizon=5)

    for idx in range(min(20, len(dataset))):  
        context, _, _ = dataset[idx]
        assert context.mean().item() == pytest.approx(0.0, abs=1e-5)


def test_normalized_context_has_std_near_one():

    df = generate_dataset(n_series=2, n_days=60, seed=0)
    dataset = SlidingWindowDataset(df, context_len=10, horizon=5)

    for idx in range(min(20, len(dataset))):
        context, _, _ = dataset[idx]
        assert context.std().item() == pytest.approx(1.0, abs=0.15)


def test_denormalization_recovers_original_values():

    n_days = 60
    context_len = 10
    horizon = 5
    df = generate_dataset(n_series=1, n_days=n_days, seed=0)
    dataset = SlidingWindowDataset(df, context_len=context_len, horizon=horizon)

    context, target, stats = dataset[0]

    context_denorm = context * stats[1] + stats[0]

    expected_values = (
        df[df["series_id"] == "series_000"]
        .sort_values("timestamp")["value"]
        .to_numpy(dtype=np.float32)
    )
    expected_context = expected_values[0:context_len]

    max_error = float(abs(context_denorm.numpy() - expected_context).max())
    assert max_error <= 1e-4, (
        f"Denormalization round-trip error too large: {max_error:.6f}"
    )


def test_target_uses_context_statistics_not_its_own():

    df = generate_dataset(n_series=1, n_days=60, seed=0)
    dataset = SlidingWindowDataset(df, context_len=10, horizon=5)

    context, target, stats = dataset[0]

    target_np = target.numpy()
    self_normalized = (target_np - target_np.mean()) / (target_np.std() + 1e-8)

    assert not np.allclose(target.numpy(), self_normalized, atol=1e-3), (
        "Target appears to be self-normalized -- possible data leakage."
    )

def test_short_series_is_skipped_without_error(capsys):

    context_len = 10
    horizon = 5
    window_size = context_len + horizon 

    df_short = generate_dataset(n_series=1, n_days=10, seed=0)
    df_short["series_id"] = "short_series"

    df_valid = generate_dataset(n_series=1, n_days=60, seed=1)
    df_mixed = pd.concat([df_short, df_valid], ignore_index=True)

    dataset = SlidingWindowDataset(
        df_mixed, context_len=context_len, horizon=horizon
    )

    expected = 60 - window_size + 1
    assert len(dataset) == expected

    captured = capsys.readouterr()
    assert "short_series" in captured.out

def test_stride_formula_matches_enumeration_across_many_inputs():

    context_len = 10
    horizon = 5
    window_size = context_len + horizon

    for n_days in range(window_size, window_size + 20):
        for stride in range(1, 6):
            df = generate_dataset(n_series=1, n_days=n_days, seed=0)
            dataset = SlidingWindowDataset(
                df, context_len=context_len, horizon=horizon, stride=stride
            )
            n_valid = n_days - window_size + 1
            expected = math.ceil(n_valid / stride)
            assert len(dataset) == expected, (
                f"Formula mismatch: n_days={n_days}, stride={stride}, "
                f"expected={expected}, got={len(dataset)}"
            )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))