import pandas as pd

def data_type(df: pd.DataFrame) -> pd.Series:
    """
    Determines the data type of each column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: A Series containing the data type of each column.
    """
    return df.dtypes