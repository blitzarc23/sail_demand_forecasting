import pandas as pd
import os
from src.model_trainer import train_and_save

# All cities you want to train for
cities = ['durgapur', 'mumbai', 'delhi', 'chennai']

# Base path to your processed files
base_path = r"C:\Users\ppran\Downloads\sail_demand_forecasting\data"

for city in cities:
    csv_path = os.path.join(base_path, f"{city}_pr.csv")

    if not os.path.exists(csv_path):
        print(f"❌ File not found for {city}: {csv_path}")
        continue

    print(f"\n📊 Training models for {city.upper()}...")
    df = pd.read_csv(csv_path, parse_dates=['date'], dayfirst=True)
    df.drop(columns=['month', 'trend_index', 'date'], inplace=True)

    for target in ['retail_sales', 'non_retail_sales']:
        train_and_save(city, df, target)
