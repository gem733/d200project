import pandas as pd

def normalise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises numeric columns in the DataFrame using z-score normalization.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df