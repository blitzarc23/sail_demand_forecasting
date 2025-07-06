import joblib

def load_model(city, target):
    path = f'models/{city}_{target}.pkl'
    return joblib.load(path)
