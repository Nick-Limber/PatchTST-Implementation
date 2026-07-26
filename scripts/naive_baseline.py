import numpy as np
import pandas as pd
from src.data_layer.synthetic import generate_dataset
from src.training.train_loop import train_validate_split

df = generate_dataset(n_series=10, n_days=500, seed=0, noise_std=2.0)
_, val_df = train_validate_split(df, train_percent=0.8)


context_len = 32
horizon     = 14
errors      = []

for series_id, group in val_df.groupby("series_id"):
    values = group.sort_values("timestamp")["value"].to_numpy()
    window_size = context_len + horizon
    for start in range(0, len(values) - window_size + 1, horizon):
        context     = values[start : start + context_len]
        target      = values[start + context_len : start + context_len + horizon]
        mean        = context.mean()
        std         = context.std() + 1e-8
        naive_pred  = np.full(horizon, context[-1])
        naive_norm  = (naive_pred  - mean) / std
        target_norm = (target      - mean) / std
        errors.append(((naive_norm - target_norm) ** 2).mean())

print(f"Naive forecast MSE (normalized): {np.mean(errors):.4f}")
print(f"PatchTST val MSE (normalized):   2.7311")
print(f"PatchTST improvement:            {((np.mean(errors) - 2.7311) / np.mean(errors) * 100):.1f}%")
