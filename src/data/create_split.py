from sklearn.model_selection import train_test_split

def create_split(df, target_col, test_size=0.1, validate_size=(1/9), random_state=42):  # validate_size is 10% of initial dataset, whihch is 1/9 of the training set after the test set is removed.
    """
    Splits the DataFrame into training and testing sets.
    
    Parameters:
    - df: The input DataFrame.
    - target_col: The name of the target column.
    - test_size: The proportion of the dataset to include in the test split.
    - validate_size: The proportion of the training set to include in the validation split.
    - random_state: Controls the randomness of the split.
    
    Returns:
    - x_train: Training features.
    - x_test: Testing features.
    - x_val: Validation features.
    - y_train: Training target values.
    - y_test: Testing target values.
    - y_val: Validation target values.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Then split the training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validate_size, random_state=random_state)

    return x_train, x_test, x_val, y_train, y_test, y_val