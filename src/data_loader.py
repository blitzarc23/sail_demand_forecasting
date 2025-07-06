import pandas as pd

def load_csv(filepath):
    """
    Load a CSV file into a pandas DataFrame.
    Args:
        filepath (str): Path to the CSV file
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(filepath, parse_dates=['date'], dayfirst=True)
    return df

def clean_data(df):
    """
    Basic cleaning: remove nulls, sort by date, etc.
    Args:
        df (pd.DataFrame): Input raw DataFrame
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.dropna()
    df = df.sort_values('date')
    return df
