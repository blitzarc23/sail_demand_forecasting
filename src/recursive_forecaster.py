import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def get_value_or_nan(df, col, date):
    match = df[df['date'] == date]
    return match[col].values[0] if not match.empty else np.nan

def generate_next_month_features(df, target, pri, sec, stk, date):
    new_row = {
        'date': date,
        'primary_price_avg': pri,
        'secondary_price_avg': sec,
        'stock_var': stk,
        'price_diff': pri - sec,
        'month_sin': np.sin(2 * np.pi * date.month / 12),
        'month_cos': np.cos(2 * np.pi * date.month / 12),
        'year': date.year,
        'quarter': date.quarter,
        'trend_index': df['trend_index'].max() + 1 if 'trend_index' in df.columns else len(df) + 1
    }

    # Generate lag features
    for lag in [1, 2, 3]:
        lag_date = date - relativedelta(months=lag)
        for col in ['retail_sales', 'non_retail_sales', 'primary_price_avg', 'secondary_price_avg', 'price_diff']:
            new_row[f'{col}_lag_{lag}'] = get_value_or_nan(df, col, lag_date)

    # Generate rolling averages
    for col in ['primary_price_avg', 'secondary_price_avg', 'retail_sales', 'non_retail_sales']:
        if col in df.columns and len(df[col].dropna()) >= 3:
            new_row[f'{col}_roll3'] = df[col].iloc[-3:].mean()

    # Custom averages
    if 'non_retail_sales' in df.columns:
        new_row['non_retail_sales_custom_avg'] = df['non_retail_sales'].iloc[-6:].mean() if len(df) >= 6 else df['non_retail_sales'].mean()

    # Add lag_12 for yearly seasonality
    lag_12_date = date - relativedelta(months=12)
    if target == 'retail_sales':
        new_row['retail_sales_lag_12'] = get_value_or_nan(df, 'retail_sales', lag_12_date)
    elif target == 'non_retail_sales':
        new_row['non_retail_sales_lag_12'] = get_value_or_nan(df, 'non_retail_sales', lag_12_date)

    return new_row
