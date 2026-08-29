import pandas as pd
import numpy as np
import torch.nn as nn
 
from src.evaluation.backtest import backtest_model, backtest_baseline, summarize_results
from src.evaluation.baselines import BASELINES
 
 
def run_comparison(
    model: nn.Module,
    test_df,
    context_len: int,
    horizon: int,
    model_name: str = "PatchTST",
    baselines: list[str] | None = None,
) -> pd.DataFrame:

        baselines = list(BASELINES.keys())
 
    rows = []
 
    print(f"Evaluating {model_name}...")
    model_results  = backtest_model(model, test_df, context_len, horizon)
    model_summary  = summarize_results(model_results)
    model_overall  = model_summary[model_summary["series_id"] == "ALL"].iloc[0]
 
    rows.append({
        "method":    model_name,
        "rmse":      model_overall["rmse"],
        "mape":      model_overall["mape"],
        "mase":      model_overall["mase"],
        "n_windows": int(model_overall["n_windows"]),
    })
 
    naive_mase = None
    for name in baselines:
        print(f"Evaluating baseline: {name}...")
        results  = backtest_baseline(name, test_df, context_len, horizon)
        summary  = summarize_results(results)
        overall  = summary[summary["series_id"] == "ALL"].iloc[0]
 
        rows.append({
            "method":    name,
            "rmse":      overall["rmse"],
            "mape":      overall["mape"],
            "mase":      overall["mase"],
            "n_windows": int(overall["n_windows"]),
        })
 
        if name == "naive":
            naive_mase = overall["mase"]
 
    comparison = pd.DataFrame(rows)
 
    if naive_mase is not None and naive_mase > 0:
        comparison["vs_naive_pct"] = (
            (naive_mase - comparison["mase"]) / naive_mase * 100
        ).round(1)
    else:
        comparison["vs_naive_pct"] = np.nan
 
    comparison = comparison.sort_values("mase").reset_index(drop=True)
 
    comparison["rmse"] = comparison["rmse"].round(3)
    comparison["mape"] = comparison["mape"].round(1)
    comparison["mase"] = comparison["mase"].round(4)
 
    return comparison
 
 
def print_report(comparison: pd.DataFrame, model_name: str = "PatchTST") -> None:
    print()
    print("=" * 70)
    print("MODEL vs BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<28} {'RMSE':>8} {'MAPE':>8} {'MASE':>8} "
          f"{'vs Naive':>10} {'Windows':>8}")
    print("-" * 70)
 
    for _, row in comparison.iterrows():
        marker = " ←" if row["method"] == model_name else ""
        vs_naive = (f"{row['vs_naive_pct']:+.1f}%"
                    if not np.isnan(row["vs_naive_pct"]) else "  n/a")
        print(
            f"{row['method']:<28} "
            f"{row['rmse']:>8.3f} "
            f"{row['mape']:>7.1f}% "
            f"{row['mase']:>8.4f} "
            f"{vs_naive:>10} "
            f"{int(row['n_windows']):>8}"
            f"{marker}"
        )
 
    print("-" * 70)
 
    model_row = comparison[comparison["method"] == model_name]
    if not model_row.empty:
        mase       = model_row["mase"].iloc[0]
        vs_naive   = model_row["vs_naive_pct"].iloc[0]
        n_windows  = int(model_row["n_windows"].iloc[0])
        direction  = "better" if vs_naive > 0 else "worse"
        print(
            f"\n{model_name} achieves MASE of {mase:.4f} -- "
            f"{abs(vs_naive):.1f}% {direction} than naive baseline "
            f"across {n_windows} evaluation windows."
        )
    print("=" * 70)
    print()
 
 
def save_report(
    comparison: pd.DataFrame,
    path: str = "outputs/reports/model_comparison.csv",
) -> None:

    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    comparison.to_csv(path, index=False)
    print(f"Comparison saved to: {path}")
