import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def select_top_features(df, target, n=10):
    # Only keep numeric columns and drop target and date/time
    numeric_df = df.select_dtypes(include='number')
    X = numeric_df.drop(columns=[target], errors='ignore')
    y = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # Drop unwanted features manually
    feature_importances = feature_importances[~feature_importances['feature'].isin(['total_sales'])]

    top_features = feature_importances.head(n)['feature'].tolist()
    return top_features, feature_importances
