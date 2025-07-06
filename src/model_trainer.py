import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_selector import select_top_features
import pandas as pd
import json

MODELS = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

# ---- Specify your custom features for Mumbai and Durgapur retail sales here ----
CUSTOM_MUMBAI_RETAIL_FEATURES = [
    "retail_sales_roll3",
    "trend_index",
    "retail_sales_lag_1",
    "price_diff_lag_3",
    "price_diff_lag_1",
    "price_diff_lag_2",
    "stock_var_lag_1",
    "stock_var_lag_3",
    "retail_sales_lag_2"
]

CUSTOM_DURGAPUR_RETAIL_FEATURES = [
     "retail_sales_roll3",
    "trend_index",
    "retail_sales_lag_1",
    "price_diff_lag_3",
    "price_diff_lag_1",
    "price_diff_lag_2",
    "stock_var_lag_1",
    "stock_var_lag_3",
    "retail_sales_lag_2"
]
def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model, best_score = None, -1
    best_name, best_metrics = '', {}

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        if r2 > best_score:
            best_model = model
            best_score = r2
            best_name = name
            best_metrics = {
                'MAE': mean_absolute_error(y_test, preds),
                'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
                'R2': r2
            }

    return best_model, best_name, best_metrics

def train_and_save(city, df, target, output_dir="models/"):
    # Use custom features for Mumbai and Durgapur retail sales
    if city.lower() == "mumbai" and target == "retail_sales":
        top_features = [f for f in CUSTOM_MUMBAI_RETAIL_FEATURES if f in df.columns]
    elif city.lower() == "durgapur" and target == "retail_sales":
        top_features = [f for f in CUSTOM_DURGAPUR_RETAIL_FEATURES if f in df.columns]
    else:
        top_features, _ = select_top_features(df, target)
        # Drop specific unwanted columns
        excluded_cols = {'non_retail_sales', target, 'retail_sales'}
        top_features = [feat for feat in top_features if feat not in excluded_cols]

    # Remove 'date' or any non-numeric column from top_features
    X = df[top_features].copy()
    if 'date' in X.columns:
        X.drop(columns='date', inplace=True)

    # Safety: Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    y = df[target]

    model, model_name, metrics = evaluate_models(X, y)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{city.lower()}_{target}.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Saved model for {city} - {target} using {model_name} with R2={metrics['R2']:.3f}")

    # --- Save filtered features to JSON ---
    feature_json_path = os.path.join(output_dir, "feature_sets.json")
    if os.path.exists(feature_json_path):
        with open(feature_json_path, "r") as f:
            feature_sets = json.load(f)
    else:
        feature_sets = {}

    feature_sets[f"{city.lower()}_{target}"] = top_features

    with open(feature_json_path, "w") as f:
        json.dump(feature_sets, f, indent=2)

    return model_path, model_name, metrics

if __name__ == "__main__":
    cities = ["delhi", "mumbai", "chennai", "durgapur"]
    targets = ["retail_sales", "non_retail_sales"]  # List of target columns

    for city in cities:
        file_path = f"data/{city}_pr.csv"
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        for target in targets:
            if target not in df.columns:
                print(f"❌ Target column '{target}' not found in {file_path}")
                continue
            train_and_save(city, df, target)