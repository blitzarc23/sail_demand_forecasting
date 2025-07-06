import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.logger import get_logger

logger = get_logger()

def simulate_future_inputs(df, months):
    logger.info("Simulating future values using linear trend...")
    logger.info(f"Simulating future inputs... df shape: {df.shape}, months: {months}")

    df = df.copy()
    df = df.sort_values("date")

    required_cols = ['primary_price_avg', 'secondary_price_avg', 'stock_var']
    for col in required_cols:
        if col not in df.columns or df[col].isnull().any():
            logger.error(f"Missing or invalid values in required column: {col}")
            return None, None, None

    # Create a time index (0, 1, 2, ..., n-1)
    df['time_index'] = np.arange(len(df))

    trends = {}
    for col in required_cols:
        X = df[['time_index']]
        y = df[col].values

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(df), len(df) + months).reshape(-1, 1)
        trend = model.predict(future_X)
        trends[col] = trend.tolist()

        logger.info(f"Simulated trend for {col}: {trend}")

    # Log lengths of generated trends before returning
    logger.info(f"Generated trends: primary({len(trends['primary_price_avg'])}), "
                f"secondary({len(trends['secondary_price_avg'])}), "
                f"stock({len(trends['stock_var'])})")

    return trends['primary_price_avg'], trends['secondary_price_avg'], trends['stock_var']
