import pandas as pd
import json
from src.model_trainer import train_and_save

CITIES = ['mumbai', 'delhi', 'chennai', 'durgapur']
TARGETS = ['retail_sales', 'non_retail_sales']
FEATURE_JSON_PATH = 'models/feature_sets.json'

feature_sets = {}

for city in CITIES:
    # Load the feature-engineered CSV
    file_path = f"data/{city}_pr.csv"
    df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)

    for target in TARGETS:
        print(f"\n🔧 Training model for {city.title()} - {target}...")
        model_path, model_name, metrics, top_features = train_and_save(city, df, target)

        # Store results in dictionary
        feature_sets[f"{city}_{target}"] = {
            "model": model_name,
            "R2": round(metrics['R2'], 4),
            "features": top_features
        }

# ✅ Save all feature sets to JSON
with open(FEATURE_JSON_PATH, "w") as f:
    json.dump(feature_sets, f, indent=4)

print(f"\n✅ All models trained. Feature sets saved to {FEATURE_JSON_PATH}")
