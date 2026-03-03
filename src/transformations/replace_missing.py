import pandas as pd

def replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces missing numeric values with the column mean.
    Creates a new column '<variable>_missing' indicating where
    missing values were originally present.

    The indicator column is only created if the variable
    actually contains missing values.
    """

    df = df.copy()

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:

        # Check if column has missing values
        if df[col].isna().any():

            # Create missing indicator (1 if missing, else 0)
            df[f"{col}_missing"] = df[col].isna().astype(int)

            # Replace missing values with mean
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)

    return df