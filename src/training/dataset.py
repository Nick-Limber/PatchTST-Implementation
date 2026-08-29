import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

from src.data_layer.schema import REQUIRED_COLUMNS, COVARIATE_PREFIX


class SlidingWindowDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        context_len: int,
        horizon: int,
        stride: int = 1,
    ):
        super().__init__()
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride

        self._index, self._series_arrays = self._build_index(df)

    # ------------------------------------------------------------------
    # PyTorch Dataset contract
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):

        series_id, start = self._index[idx]
        values = self._series_arrays[series_id]

        context = values[start : start + self.context_len]
        target = values[
            start + self.context_len : start + self.context_len + self.horizon
        ]

        mean = context.mean()
        std = context.std() + 1e-8

        context_norm = (context - mean) / std
        target_norm = (target - mean) / std

        return (
            torch.tensor(context_norm, dtype=torch.float32),
            torch.tensor(target_norm, dtype=torch.float32),
            torch.tensor([mean, std], dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self, df: pd.DataFrame):

        index = []
        series_arrays = {}
        window_size = self.context_len + self.horizon

        for series_id, group in df.groupby("series_id"):

            values = (
                group
                .sort_values("timestamp")["value"]
                .to_numpy(dtype=np.float32)
            )

            if len(values) < window_size:

                print(
                    f"Warning: series '{series_id}' has {len(values)} timesteps "
                    f"but context_len + horizon = {window_size}. Skipping."
                )
                continue

            n_windows = sliding_window_view(
                values, window_shape=window_size
            ).shape[0]

            start_positions = np.arange(0, n_windows, self.stride)

            for start in start_positions.tolist():
                index.append((series_id, start))

            series_arrays[series_id] = values

        return index, series_arrays


# ---------------------------------------------------------------------------
# Quick manual check -- run with: python -m src.training.dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data_layer.synthetic import generate_dataset

    # Generate a small synthetic dataset.
    df = generate_dataset(n_series=3, n_days=60, seed=0)
    print(f"Raw dataframe: {len(df)} rows, columns: {list(df.columns)}")
    print()

    context_len = 10
    horizon = 5
    dataset = SlidingWindowDataset(df, context_len=context_len, horizon=horizon)

    print(f"Total windows: {len(dataset)} (expected {3 * (60 - context_len - horizon + 1)})")
    print()

    context, target, stats = dataset[0]
    print(f"context shape : {context.shape}  (expected [{context_len}])")
    print(f"target shape  : {target.shape}   (expected [{horizon}])")
    print(f"stats shape   : {stats.shape}    (expected [2])")
    print(f"context mean  : {context.mean():.6f}  (expected ≈ 0.0 after normalization)")
    print(f"context std   : {context.std():.6f}   (expected ≈ 1.0 after normalization)")
    print(f"stored mean   : {stats[0]:.4f}")
    print(f"stored std    : {stats[1]:.4f}")
    print()

    context_denorm = context * stats[1] + stats[0]
    raw_values = df[df["series_id"] == "series_000"].sort_values("timestamp")["value"].values
    expected_context = raw_values[0:context_len]
    max_diff = float(np.abs(context_denorm.numpy() - expected_context).max())
    print(f"Denormalization round-trip max error: {max_diff:.8f} (expected ≈ 0.0)")
    print()

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_context, batch_target, batch_stats = next(iter(loader))
    print(f"Batch context shape : {batch_context.shape}  (expected [8, {context_len}])")
    print(f"Batch target shape  : {batch_target.shape}   (expected [8, {horizon}])")
    print(f"Batch stats shape   : {batch_stats.shape}    (expected [8, 2])")