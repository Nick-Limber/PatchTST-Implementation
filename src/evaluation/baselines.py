 
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
 
 
def naive_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    last_value = context[-1]
    return np.full(horizon, last_value)
 
 
def seasonal_naive_forecast(
    context: np.ndarray,
    horizon: int,
    season_length: int = 7,
) -> np.ndarray:
    if len(context) < season_length:
        return naive_forecast(context, horizon)
 
    forecast = np.empty(horizon)
    for h in range(horizon):
        lookback_idx = len(context) - season_length + (h % season_length)
        forecast[h]  = context[lookback_idx]
 
    return forecast
 
 
def moving_average_forecast(
    context: np.ndarray,
    horizon: int,
    window: int = 7,
) -> np.ndarray:
    effective_window = min(window, len(context))
    mean_value       = context[-effective_window:].mean()
    return np.full(horizon, mean_value)
 
 
def exponential_smoothing_forecast(
    context: np.ndarray,
    horizon: int,
) -> np.ndarray:
    try:
        model        = SimpleExpSmoothing(context, initialization_method="estimated")
        fitted       = model.fit(optimized=True)
        last_smoothed = fitted.fittedvalues[-1]
        return np.full(horizon, last_smoothed)
    except Exception:
        return naive_forecast(context, horizon)
 
BASELINES = {
    "naive":                 naive_forecast,
    "seasonal_naive":        seasonal_naive_forecast,
    "moving_average":        moving_average_forecast,
    "exponential_smoothing": exponential_smoothing_forecast,
}
 
if __name__ == "__main__":
    np.random.seed(42)
    context = np.array([
        80, 85, 90, 95, 100, 115, 70,   # week 1: Mon-Sun
        82, 87, 92, 97, 102, 118, 72,   # week 2
        84, 89, 94, 99, 104, 120, 74,   # week 3
        86, 91, 96, 101, 106, 122, 76,  # week 4
    ], dtype=float)
 
    horizon = 7
    print(f"Context (last 7 days): {context[-7:]}")
    print(f"Horizon: {horizon} days\n")
 
    for name, fn in BASELINES.items():
        forecast = fn(context, horizon)
        print(f"{name:25s}: {np.round(forecast, 1)}")
 
    print("\nExpected seasonal_naive to match context[-7:]:")
    print(f"  context[-7:] = {context[-7:]}")
    seasonal = seasonal_naive_forecast(context, horizon)
    assert np.allclose(seasonal, context[-7:]), "Seasonal naive mismatch"
    print("  seasonal_naive matches context[-7:] -- correct")
 
    print("\nExpected naive to be constant [76, 76, ...]:")
    naive = naive_forecast(context, horizon)
    assert np.all(naive == context[-1]), "Naive mismatch"
    print(f"  naive = {naive} -- correct")
 
    print("\nAll manual checks passed.")
