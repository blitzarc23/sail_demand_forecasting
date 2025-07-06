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
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)