import pandas as pd

def missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Identifies the number of missing values in each column of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Series containing the count of missing values for each column.
    """
    return df.isnull().sum()
