import pandas as pd

def convert_strings(df, columns):
    """
    Convert specified columns in a DataFrame from strings to ints or floats.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to convert.
    
    Returns:
    pd.DataFrame: DataFrame with specified columns converted to numeric types.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df