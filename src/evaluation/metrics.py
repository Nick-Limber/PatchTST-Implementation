import numpy as np

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    
    error = predicted - actual
    mse = np.mean(error ** 2)
    rmse = float(np.sqrt(mse))

    return rmse 


def mape(actual: np.ndarray, predicted: np.ndarray):

    epsilon     = np.finfo(float).eps
    actual_safe = np.where(actual == 0, epsilon, actual)

    return float(np.mean(np.abs(predicted - actual_safe) / actual_safe) * 100)


def mase(actual: np.ndarray, predicted: np.ndarray, naive_forecast: np.ndarray) -> float:
    model_error = np.mean(np.abs(predicted       - actual))
    naive_error = np.mean(np.abs(naive_forecast  - actual))
    
    return float(model_error / (naive_error + np.finfo(float).eps))

def compute_all_metrics(    actual: np.ndarray, predicted: np.ndarray, naive_forecast: np.ndarray) -> dict:

    return {
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "mase": mase(actual, predicted, naive_forecast) 
        }
