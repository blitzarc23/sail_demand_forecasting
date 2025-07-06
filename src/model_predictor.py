import joblib
import pandas as pd

def load_model(model_path):
    """
    Load a trained model from disk.
    """
    return joblib.load(model_path)

def make_predictions(model, df, features):
    """
    Use the model to predict target values.
    """
    X = df[features]
    preds = model.predict(X)
    return preds
