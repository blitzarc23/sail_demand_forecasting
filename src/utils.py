import os
import pandas as pd
from dateutil.relativedelta import relativedelta
import pandas as pd

def generate_monthly_dates(start_date, months):
    start_date = pd.to_datetime(start_date)  # ✅ ensure it's datetime
    return [start_date + relativedelta(months=i) for i in range(months)]

def get_model_path(city, target):
    return os.path.join("models", f"{city}_{target}.pkl")
def format_dates(df):
    # Example implementation
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['date'] = df['date'].dt.strftime('%d-%m-%Y')
    return df


def validate_city_and_target(city, target):
    valid_cities = ['mumbai', 'delhi', 'chennai', 'durgapur']
    valid_targets = ['retail_sales', 'non_retail_sales']
    
    if city.lower() not in valid_cities:
        raise ValueError(f"Invalid city '{city}'. Choose from {valid_cities}.")
    
    if target.lower() not in valid_targets:
        raise ValueError(f"Invalid sales type '{target}'. Choose from {valid_targets}.")
