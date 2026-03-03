import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Data loaded from the CSV.
    """

    df = pd.read_csv(file_path)
    return df
