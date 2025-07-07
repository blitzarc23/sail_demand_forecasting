
# TMT Rebar Demand Forecasting Web Application

## Overview

This project is a web-based forecasting tool developed to predict monthly TMT rebar sales (retail and non-retail) across four major Indian cities: Mumbai, Delhi, Chennai, and Durgapur. The application leverages historical pricing and sales data to simulate future values and generate forecasts using machine learning models. It supports city- and segment-specific predictions with an intuitive user interface.

## Features

- Interactive web interface built using Flask
- Automatic feature engineering and selection per city and sales segment
- Recursive multi-step forecasting for up to 6 months
- Trend-based simulation of future prices and stock variations
- Downloadable forecast outputs
- Visualizations for sales forecasts and feature importances
- Lightweight and modular backend for easy updates

## Project Structure

```
project/
│
├── app.py                      # Flask application entry point
├── templates/
│   ├── index.html              # Home page
│   └── result.html             # Forecast results page
├── static/
│   ├── style.css               # Optional CSS file
│   └── bg.jpg                  # Background image
├── models/                     # Pre-trained machine learning models
│   └── *.pkl
├── data/
│   └── city_pr.csv             # Feature-engineered datasets (one per city)
├── src/
│   ├── forecast_utils.py       # Forecasting logic
│   ├── feature_simulator.py    # Price and stock simulation
│   ├── recursive_forecaster.py # Recursive prediction loop
│   └── feature_selector.py     # Feature selection per city and segment
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Flask for the web interface
- HTML, CSS (basic styling)
- Matplotlib / Plotly for visualizations (optional)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/tmt-forecasting.git
   cd tmt-forecasting
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Launch the Flask app:
   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage

1. Select the city and sales segment (Retail or Non-retail).
2. Click "Forecast" to generate a 6-month sales prediction.
3. View forecast plots and tables on the result page.
4. Optionally, download the predicted data for offline use.

## Data

- Monthly primary and secondary price averages (2022–2025)
- Monthly retail and non-retail sales volumes per city
- Derived features: lags, rolling averages, price differences, stock variation trends

## License

This project is for academic and research purposes. Please contact the author before using it for commercial applications.

## Contributors

- [Pranshu Tijil]
- Steel Authority of India (SAIL) – Data Source
