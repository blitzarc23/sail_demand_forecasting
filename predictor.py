import pandas as pd
import joblib
from src.feature_engineering import prepare_input_row

def forecast_next_months(city, target, months=6):
    model_path = f"models/{city}_{target}.pkl"
    model = joblib.load(model_path)

    # Load the full city dataset
    df = pd.read_csv(f"data/{city}_pr.csv", parse_dates=['date'], dayfirst=True)
    df = df.sort_values('date').reset_index(drop=True)

    predictions = []
    dates = []

    for i in range(months):
        # Use last known values to prepare next input row
        last_row = prepare_input_row(city, df['primary_price_avg'].iloc[-1], df['secondary_price_avg'].iloc[-1])
        features = last_row.drop(columns=['date', 'retail_sales', 'non_retail_sales'])

        pred = model.predict(features)[0]
        predictions.append(round(pred, 2))
        forecast_date = last_row['date'].values[0]
        dates.append(pd.to_datetime(forecast_date))

        # Append the predicted sales back to df for the next iteration
        if target == 'retail_sales':
            last_row['retail_sales'] = pred
        else:
            last_row['non_retail_sales'] = pred

        df = pd.concat([df, last_row], ignore_index=True)

    forecast_df = pd.DataFrame({
        'date': dates,
        'predicted_sales': predictions
    })

    return forecast_df
