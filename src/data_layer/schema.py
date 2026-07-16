"""
 Schema for sales time series. Used to validate data. Includes required columns and map all covariate columns to a single standardize naming convention
 
    timestamp   : datetime64[ns]   required
    series_id   : str              required (e.g. "store_1_food")
    value       : float            required, the target being forecast
    <covariates>: float / category, optional (e.g. "sell_price", "is_holiday")
"""

REQUIRED_COLUMNS = ["timestamp", "series_id", "value"]
COVARIATE_PREFIX = "covariate_"
DEFAULT_FREQUENCY = "D"