import pandas as pd
from dateutil.relativedelta import relativedelta
from src.feature_simulator import simulate_future_inputs
from src.model_loader import load_model
from src.utils import format_dates, generate_monthly_dates
from src.logger import get_logger
from src.recursive_forecaster import generate_next_month_features

logger = get_logger()

def forecast_next_months(city, target, months=6):
    logger.info("Entered forecast_next_months()")
    
    file_path = f"data/{city}_pr.csv"
    logger.info(f"Reading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)
        df.columns = df.columns.str.strip()
        if 'date' not in df.columns:
            logger.error(f"Expected 'date' column, found: {df.columns.tolist()}")
            return None
        df.sort_values('date', inplace=True)
        logger.info(f"Max date value: {df['date'].max()}")
    except Exception as e:
        logger.error(f"Failed to read or format data: {e} | File path was: {file_path}")
        return None

    try:
        latest_date = pd.to_datetime(df['date'].max())
        logger.info(f"Last date in dataset: {latest_date.strftime('%d-%m-%Y')}")

        logger.info("Generating future dates...")
        future_dates = generate_monthly_dates(latest_date + relativedelta(months=1), months)

        pri_trend, sec_trend, stk_trend = simulate_future_inputs(df, months)
        model = load_model(city, target)
        feature_names = model.feature_names_in_

        preds = []
        working_df = df.copy()

        for i in range(months):
            new_date = future_dates[i]
            new_row = generate_next_month_features(working_df, target, pri_trend[i], sec_trend[i], stk_trend[i], new_date)
            working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)

        # Forward-fill missing values with last known value after all simulated rows are appended
        working_df.ffill(inplace=True)

        # Predict for only the future months
        forecast_rows = working_df.iloc[-months:]
        for i, row in forecast_rows.iterrows():
            input_df = pd.DataFrame([row])
            missing_features = set(feature_names) - set(input_df.columns)
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                for feature in missing_features:
                    # Use last known value from working_df for each missing feature
                    if feature in working_df.columns:
                        input_df[feature] = working_df[feature].iloc[-months-1]  # last known value before forecast period
                        logger.warning(f"Setting missing feature '{feature}' to last known value {working_df[feature].iloc[-months-1]}")
                    else:
                        input_df[feature] = None

            X = pd.DataFrame([row], columns=feature_names)

            if X.isnull().any().any():
                logger.error(f"Generated input for {row['date'].strftime('%b-%Y')} contains NaN. Skipping this prediction.")
                continue

            y_pred = model.predict(X)[0]
            preds.append({'date': row['date'], f'predicted_{target}': y_pred})
            logger.info(f"Forecasted {target} for {city} on {row['date'].strftime('%b-%Y')}: {y_pred:.2f}")
            logger.info("Forecasting completed successfully.")
        forecast_df = pd.DataFrame(preds)
        return forecast_df

    except Exception as e:
        logger.error(f"Error while forecasting: {e} | File path was: {file_path}")
        return None