import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Callable
 
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.baselines import naive_forecast, BASELINES
 
def split_three_way(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_frames = []
    val_frames   = []
    test_frames  = []
 
    for series_id in df["series_id"].unique():
        series_df = (
            df[df["series_id"] == series_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        n          = len(series_df)
        train_end  = int(n * train_frac)
        val_end    = int(n * (train_frac + val_frac))
 
        train_frames.append(series_df.iloc[:train_end])
        val_frames.append(series_df.iloc[train_end:val_end])
        test_frames.append(series_df.iloc[val_end:])
 
    return (
        pd.concat(train_frames,  ignore_index=True),
        pd.concat(val_frames,    ignore_index=True),
        pd.concat(test_frames,   ignore_index=True),
    )
 
def _evaluate_windows(
    forecast_fn: Callable,
    df: pd.DataFrame,
    context_len: int,
    horizon: int,
) -> list[dict]:
 
    results      = []
    window_size  = context_len + horizon
 
    for series_id, group in df.groupby("series_id"):
        values = (
            group
            .sort_values("timestamp")["value"]
            .to_numpy(dtype=np.float64)
        )
 
        if len(values) < window_size:
            print(
                f"Warning: series '{series_id}' has {len(values)} timesteps "
                f"but context_len + horizon = {window_size}. Skipping."
            )
            continue
 
        for start in range(0, len(values) - window_size + 1, horizon):
            context = values[start : start + context_len]
            actual  = values[start + context_len : start + window_size]
 
            predicted = forecast_fn(context)
 
            naive_pred = naive_forecast(context, horizon)
 
            window_metrics = compute_all_metrics(actual, predicted, naive_pred)
            window_metrics["series_id"] = series_id
            window_metrics["origin"]    = start
 
            results.append(window_metrics)
 
    return results
 
def backtest_model(
    model: nn.Module,
    test_df: pd.DataFrame,
    context_len: int,
    horizon: int,
) -> list[dict]:

    model.eval()
 
    def model_forecast(context: np.ndarray) -> np.ndarray:
        mean = context.mean()
        std  = context.std() + 1e-8   # epsilon prevents divide by zero
 
        context_norm = (context - mean) / std
 
        x = torch.tensor(context_norm, dtype=torch.float32).unsqueeze(0)
 
        with torch.no_grad():
            prediction_norm = model(x)
 
        prediction_norm = prediction_norm.squeeze(0).numpy()
 
        return prediction_norm * std + mean
 
    return _evaluate_windows(model_forecast, test_df, context_len, horizon)
 
 
def backtest_baseline(
    baseline_name: str,
    test_df: pd.DataFrame,
    context_len: int,
    horizon: int,
    **baseline_kwargs,
) -> list[dict]:
 
 
    if baseline_name not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{baseline_name}'. "
            f"Available: {list(BASELINES.keys())}"
        )
 
    baseline_fn = BASELINES[baseline_name]
 
    def forecast_fn(context: np.ndarray) -> np.ndarray:
        return baseline_fn(context, horizon, **baseline_kwargs)
 
    results = _evaluate_windows(forecast_fn, test_df, context_len, horizon)
    for r in results:
        r["method"] = baseline_name
 
    return results
 
def summarize_results(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
 
    per_series = (
        df.groupby("series_id")[["rmse", "mape", "mase"]]
        .agg(["mean", "count"])
    )
    per_series.columns = ["rmse", "rmse_count", "mape", "mape_count",
                          "mase", "mase_count"]
    per_series = per_series[["rmse", "mape", "mase", "rmse_count"]].copy()
    per_series.columns = ["rmse", "mape", "mase", "n_windows"]
    per_series = per_series.reset_index()
 
    overall = pd.DataFrame([{
        "series_id": "ALL",
        "rmse":      df["rmse"].mean(),
        "mape":      df["mape"].mean(),
        "mase":      df["mase"].mean(),
        "n_windows": len(df),
    }])
 
    return pd.concat([per_series, overall], ignore_index=True)
 
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
 
    from src.data_layer.synthetic import generate_dataset
    from src.models.patchtst.model import PatchTST
 
    print("=" * 60)
    print("Backtest end-to-end check")
    print("=" * 60)
 
    context_len   = 32
    horizon       = 7
    n_series      = 4
    n_days        = 300

    df = generate_dataset(n_series=n_series, n_days=n_days, seed=0)
    train_df, val_df, test_df = split_three_way(df, train_frac=0.7, val_frac=0.15)
 
    print(f"Total days per series : {n_days}")
    print(f"Train days per series : {len(train_df) // n_series}")
    print(f"Val days per series   : {len(val_df)   // n_series}")
    print(f"Test days per series  : {len(test_df)  // n_series}")
    print()
 
    model = PatchTST(
        context_len=context_len, patch_len=8, stride=4,
        d_model=32, n_heads=2, n_layers=2, d_feedforward=64,
        dropout=0.0, horizon=horizon,
    )
    print("Model built (untrained -- just testing the pipeline)")
 
    print("\nRunning model backtest...")
    model_results = backtest_model(model, test_df, context_len, horizon)
    print(f"Evaluation windows: {len(model_results)}")
 
    print("\nRunning baseline backtests...")
    baseline_results = {}
    for name in BASELINES:
        baseline_results[name] = backtest_baseline(
            name, test_df, context_len, horizon
        )
 
    model_summary = summarize_results(model_results)
    print(model_summary.to_string(index=False))


    print("\nBaseline results:")
    for name, results in baseline_results.items():
        summary = summarize_results(results)
        overall = summary[summary["series_id"] == "ALL"].iloc[0]
        print(f"  {name:25s}: RMSE={overall['rmse']:.3f}  "
              f"MAPE={overall['mape']:.1f}%  MASE={overall['mase']:.3f}")


    naive_mase    = naive_summary[naive_summary["series_id"] == "ALL"]["mase"].iloc[0]
    assert abs(naive_mase - 1.0) < 0.01, \
        f"Naive MASE should be 1.0 by definition, got {naive_mase:.4f}"
    print(f"\nSanity check: naive MASE = {naive_mase:.4f} (expected 1.0) ✓")
    print("\nAll checks passed.")
