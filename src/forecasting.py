import pandas as pd
import joblib
import os

def forecast_next_months(city, target, months=6):
    model_path = f"models/{city}_{target}.pkl"
    data_path = f"data/{city}_pr.csv"

    # ✅ Load model using joblib (not pickle)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)

    # ✅ Load and sort data
    df = pd.read_csv(data_path, parse_dates=['date'])
    df = df.sort_values('date')

    # ✅ Start from last row
    last_row = df.iloc[-1:].copy()
    future_rows = []

    for _ in range(months):
        last_date = last_row['date'].values[0]
        next_date = pd.to_datetime(last_date) + pd.DateOffset(months=1)
        new_row = last_row.copy()
        new_row['date'] = next_date
        new_row['month'] = next_date.month

        # ✅ Update lag features
        if target == 'retail_sales':
            for i in range(3, 0, -1):
                new_row[f'retail_sales_lag_{i}'] = (
                    last_row[f'retail_sales_lag_{i-1}'].values[0]
                    if i > 1 else last_row['retail_sales'].values[0]
                )
        else:
            for i in range(3, 0, -1):
                new_row[f'non_retail_sales_lag_{i}'] = (
                    last_row[f'non_retail_sales_lag_{i-1}'].values[0]
                    if i > 1 else last_row['non_retail_sales'].values[0]
                )

        # ✅ Drop extra columns not used in training
        features = model.feature_names_in_
        new_row_features = new_row[features]

        # ✅ Predict and update target
        y_pred = model.predict(new_row_features)[0]
        new_row[target] = y_pred

        # ✅ Save predicted result
        future_rows.append({
            'date': next_date,
            f'predicted_{target}': y_pred
        })

        # ✅ Use updated row for next iteration
        last_row = new_row.copy()

    # ✅ Return forecasted DataFrame
    forecast_df = pd.DataFrame(future_rows)
    return forecast_df
