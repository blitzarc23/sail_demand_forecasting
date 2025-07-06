import pandas as pd

def generate_features_for_month(df):
    last_date = df['date'].max()
    next_date = (last_date + pd.DateOffset(months=1)).replace(day=1)

    # Create the new forecast row based on latest known values
    row = {
        'date': next_date,
        'primary_price_avg': df['primary_price_avg'].iloc[-1],
        'secondary_price_avg': df['secondary_price_avg'].iloc[-1],
        'price_diff': df['primary_price_avg'].iloc[-1] - df['secondary_price_avg'].iloc[-1]
    }

    for col in ['retail_sales', 'non_retail_sales']:
        row[col] = df[col].iloc[-1] if col in df.columns else 0

    # Append the new row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)

    # Lag and rolling features
    df['retail_sales_lag_1'] = df['retail_sales'].shift(1)
    df['retail_sales_lag_2'] = df['retail_sales'].shift(2)
    df['retail_sales_lag_3'] = df['retail_sales'].shift(3)
    df['retail_sales_roll3'] = df['retail_sales'].rolling(3).mean()

    df['non_retail_sales_lag_1'] = df['non_retail_sales'].shift(1)
    df['non_retail_sales_lag_2'] = df['non_retail_sales'].shift(2)
    df['non_retail_sales_lag_3'] = df['non_retail_sales'].shift(3)
    df['non_retail_sales_roll3'] = df['non_retail_sales'].rolling(3).mean()

    df['price_diff_lag_1'] = df['price_diff'].shift(1)
    df['price_diff_lag_2'] = df['price_diff'].shift(2)
    df['price_diff_lag_3'] = df['price_diff'].shift(3)

    # Custom average (with fallback)
    df['non_retail_sales_custom_avg'] = df['non_retail_sales'].rolling(3, center=True).mean()
    if pd.isna(df['non_retail_sales_custom_avg'].iloc[-1]):
        df.at[df.index[-1], 'non_retail_sales_custom_avg'] = df['non_retail_sales'].iloc[-4:-1].mean()

    # Separate future row and clean historical data
    historical_df = df.iloc[:-1].dropna()
    future_row = df.iloc[[-1]].copy()

    # Fill any remaining NaNs in future row with ffill/bfill or 0
    future_row.fillna(method='ffill', inplace=True)
    future_row.fillna(method='bfill', inplace=True)
    future_row.fillna(0, inplace=True)

    # Combine back together
    df_cleaned = pd.concat([historical_df, future_row], ignore_index=True)
    
    return df_cleaned, future_row
