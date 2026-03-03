import pandas as pd

def one_hot_encode(df, column):
    """
    One-hot encode a column.
    - Column names become lowercase
    - Spaces replaced with underscores
    """

    # Create dummy variables
    dummies = pd.get_dummies(df[column])

    # Clean column names
    dummies.columns = (
        dummies.columns
        .str.lower()
        .str.replace(" ", "_")
    )

    # Drop original column
    df = df.drop(columns=[column])

    # Add cleaned dummy columns
    df = pd.concat([df, dummies], axis=1)

    return df