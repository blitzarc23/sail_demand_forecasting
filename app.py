from flask import Flask, render_template, request, send_file
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import io
from src.forecast_utils import forecast_next_months
from src.utils import validate_city_and_target
from src.logger import get_logger
import sys
import os
import numpy as np
from src.google_sheets_utils import read_sheet_as_df, write_df_to_sheet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

logger = get_logger()

app = Flask(__name__)

CITY_TO_REGION = {
    "mumbai": "Western Region",
    "delhi": "Northern Region",
    "chennai": "Southern Region",
    "durgapur": "Eastern Region",
    "india": "India"
}

@app.route('/')
def home():
    city = request.args.get('city')
    next_date_msg = ""
    if city:
        file_path = f"data/{city}_pr.csv"
        try:
            df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)
            last_date = df['date'].max()
            next_month = (last_date + pd.DateOffset(months=1)).strftime('%B %Y')
            next_date_msg = f"Please Input Values for Next Month - {next_month}"
        except Exception:
            next_date_msg = ""
    return render_template('index.html', next_date_msg=next_date_msg)
    
import logging

def get_last_date(city):
    file_path = f"data/{city}_pr.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)
        last_date = df['date'].max()
        logger.info(f"Successfully read {file_path}. Last date: {last_date.strftime('%d-%m-%Y')}")
        return last_date
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        city = request.form.get('city')
        target = request.form.get('target')
        months = int(request.form.get('months'))

        if not city or not target or not months:
            return render_template('index.html', error="All fields are required.")

        regions = ['mumbai', 'delhi', 'chennai', 'durgapur']
        region_label = CITY_TO_REGION.get(city, city.title())
          # --- All India CSV completeness check ---
        if city == "india":
            last_dates = {region: get_last_date(region) for region in regions}
            if None in last_dates.values():
                return render_template('index.html', error="Could not read all region CSV files. Check logs.")
            if len(set(last_dates.values())) > 1:
                logger.warning(f"CSV files not complete for all regions. Last dates: {last_dates}")
                return render_template('index.html', error="All region CSVs must be filled to the same date for Whole India prediction. Please update missing months.")
            logger.info(f"All region CSVs complete for Whole India prediction. Last date: {list(last_dates.values())[0].strftime('%d-%m-%Y')}")

        # Handle "Whole India" selection
        if city == "india":
            region_forecasts = []
            city_names = ['mumbai', 'delhi', 'chennai', 'durgapur']
            for region in city_names:
                retail_df = forecast_next_months(region, "retail_sales", months)
                non_retail_df = forecast_next_months(region, "non_retail_sales", months)
                # Try to forecast stock_var, handle missing model gracefully
                try:
                    stock_var = forecast_next_months(region, "stock_var", months)
                except Exception:
                    stock_var = None
                merged = pd.merge(retail_df, non_retail_df, on="date", how="inner")
                if stock_var is not None and 'predicted_stock_var' in stock_var.columns:
                    merged = pd.merge(merged, stock_var[['date', 'predicted_stock_var']], on="date", how="left")
                else:
                    merged['predicted_stock_var'] = 0  # or np.nan
                merged['city'] = region
                merged['predicted_total_sales'] = merged['predicted_retail_sales'] + merged['predicted_non_retail_sales']
                region_forecasts.append(merged)

            all_cities_df = pd.concat(region_forecasts, ignore_index=True)
            india_months = np.sort(all_cities_df['date'].unique())
            india_results = []
            city_month_df = all_cities_df.copy()

            for idx, month in enumerate(india_months):
                month_df = city_month_df[city_month_df['date'] == month]
                total_sales_sum = month_df['predicted_total_sales'].sum()
                stock_var_sum = month_df['predicted_stock_var'].sum() if 'predicted_stock_var' in month_df.columns else 0

                # Rule 1: If total_sales < 275k, add to next month's stock_var
                if total_sales_sum < 275_000 and idx + 1 < len(india_months):
                    next_month = india_months[idx + 1]
                    add_val = float(275_000 - total_sales_sum)
                    city_month_df.loc[city_month_df['date'] == next_month, 'predicted_stock_var'] = (
                        city_month_df.loc[city_month_df['date'] == next_month, 'predicted_stock_var'].astype(float) + add_val
                    )

                # Rule 2: If stock_var > 250k, distribute excess to non_retail_sales in ratio of total_sales
                if stock_var_sum > 250_000:
                    excess = stock_var_sum - 250_000
                    ratios = month_df['predicted_total_sales'] / total_sales_sum if total_sales_sum > 0 else 0
                    for city_idx, row in month_df.iterrows():
                        add_val = excess * ratios.loc[city_idx]
                        city_month_df.loc[city_idx, 'predicted_non_retail_sales'] += add_val

                india_results.append({
                    'date': month,
                    'predicted_total_sales': city_month_df[city_month_df['date'] == month]['predicted_total_sales'].sum(),
                    'predicted_retail_sales': city_month_df[city_month_df['date'] == month]['predicted_retail_sales'].sum(),
                    'predicted_non_retail_sales': city_month_df[city_month_df['date'] == month]['predicted_non_retail_sales'].sum(),
                })

            forecast_df = pd.DataFrame(india_results)
            y_col = f'predicted_{target}' if target != "total_sales" else "predicted_total_sales"
            plot_title = f"{months}-Month Forecast for {target.replace('_', ' ').title()} in {region_label}"

        else:
            # Handle total_sales for a single region
            if target == "total_sales":
                retail_df = forecast_next_months(city, "retail_sales", months)
                non_retail_df = forecast_next_months(city, "non_retail_sales", months)
                forecast_df = pd.merge(retail_df, non_retail_df, on="date", how="inner")
                forecast_df['predicted_total_sales'] = forecast_df['predicted_retail_sales'] + forecast_df['predicted_non_retail_sales']
                y_col = 'predicted_total_sales'
                plot_title = f"{months}-Month Forecast for Total Sales in {region_label}"
            else:
                forecast_df = forecast_next_months(city, target, months)
                y_col = f'predicted_{target}'
                plot_title = f"{months}-Month Forecast for {target.replace('_', ' ').title()} in {region_label}"

        if forecast_df is None or forecast_df.empty:
            return render_template('index.html', error="Forecast failed or returned no data.")

        # Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df[y_col],
            mode='lines+markers',
            name=f'Forecasted {target}'
        ))
        fig.update_layout(title=plot_title,
                          xaxis_title="Date",
                          yaxis_title="Sales")

        plot_html = pio.to_html(fig, full_html=False)

        # CSV export
        csv_buffer = io.StringIO()
        forecast_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        return render_template('result.html',
                               plot_html=plot_html,
                               csv_data=csv_data,
                               filename=f"{region_label.replace(' ', '_').lower()}_{target}_forecast.csv")

    except Exception as e:
        logger.info(f"User submitted forecast request: city={city}, target={target}, months={months}")
        logger.error(f"Error while forecasting: {str(e)}")
        return render_template('index.html', error="Something went wrong. Check logs.")
@app.route('/feature_engineer', methods=['POST'])
def feature_engineer():
    try:
        city = request.form['city']
        primary_price_avg = float(request.form['primary_price_avg'])
        secondary_price_avg = float(request.form['secondary_price_avg'])
        stock_var = float(request.form['stock_var'])
        retail_sales = float(request.form.get('retail_sales', 0))
        non_retail_sales = float(request.form.get('non_retail_sales', 0))

        # Use Google Sheets instead of CSV
        df = read_sheet_as_df(f"{city}_pr")
        df = df.sort_values('date').reset_index(drop=True)

        # Compute next date
        last_date = pd.to_datetime(df['date'].max())
        next_date = last_date + pd.DateOffset(months=1)

        # Append new row
        new_row = {
            'date': next_date,
            'primary_price_avg': primary_price_avg,
            'secondary_price_avg': secondary_price_avg,
            'stock_var': stock_var,
            'retail_sales': retail_sales,
            'non_retail_sales': non_retail_sales,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Feature engineering (same as before)
        df['price_diff'] = df['primary_price_avg'] - df['secondary_price_avg']
        df['total_sales'] = df['retail_sales'] + df['non_retail_sales']
        for col in ['primary_price_avg', 'secondary_price_avg', 'stock_var', 'retail_sales', 'non_retail_sales', 'price_diff']:
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        df['primary_price_avg_roll_3'] = df['primary_price_avg'].rolling(3, min_periods=1, center=True).mean()
        df['secondary_price_avg_roll_3'] = df['secondary_price_avg'].rolling(3, min_periods=1, center=True).mean()
        df['retail_sales_roll_3'] = df['retail_sales'].rolling(3, min_periods=1, center=True).mean()
        df['non_retail_sales_roll_3'] = df['non_retail_sales'].rolling(3, min_periods=1, center=True).mean()
        df['trend_index'] = range(1, len(df) + 1)
        df['month_number'] = pd.to_datetime(df['date'], dayfirst=True).dt.month
        n = len(df)
        non_retail = df['non_retail_sales'].copy()
        if n >= 2:
            non_retail.iloc[-1] = non_retail.iloc[-2]
        df['non_retail_sales_custom_avg'] = non_retail.rolling(3, min_periods=1, center=True).mean()
        df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.strftime('%d-%m-%Y')

        # Save back to Google Sheet
        write_df_to_sheet(df, f"{city}_pr")

        next_next_date = (pd.to_datetime(next_date) + pd.DateOffset(months=1)).strftime('%B %Y')
        return render_template('index.html', message="Row added and features updated!", next_date_msg=f"Please Input Values for Next Month - {next_next_date}")

    except Exception as e:
        logger.error(f"Error in feature_engineer: {e}")
        return render_template('index.html', error="Something went wrong while updating features. Check logs.")
@app.route('/download', methods=['POST'])
def download():
    csv_data = request.form['csv_data']
    filename = request.form['filename']
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

import os
@app.route("/test_sheets")
def test_sheets():
    try:
        df = read_sheet_as_df("mumbai_pr")  # use a valid sheet tab
        return f"✅ Loaded {len(df)} rows."
    except Exception as e:
        return f"❌ Failed: {e}"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
