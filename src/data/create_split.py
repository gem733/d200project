from sklearn.model_selection import train_test_split

def create_split(df, target_col, test_size=0.1, validate_size=(2/9), validate_ensemble_size=(1/2), random_state=42):  # total validate size is 20% of initial dataset, whihch is 2/9 of the training set after the test set is removed. The validation set is then split in half.
    """
    Splits the DataFrame into training and testing sets.
    
    Parameters:
    - df: The input DataFrame.
    - target_col: The name of the target column.
    - test_size: The proportion of the dataset to include in the test split.
    - validate_size: The proportion of the training set to include in the validation split.
    - random_state: Controls the randomness of the split.
    
    Returns:
    - x_train: Training set.
    - x_test: Testing set.
    - x_val: Validation set.
    - x_val_ens: Validation set for ensemble model.
    - y_train: Training target values.
    - y_test: Testing target values.
    - y_val: Validation target values.
    - y_val_ens: Validation target values for ensemble model.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Split remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validate_size, random_state=random_state)
    
    # Split validation set into validation and ensemble validation sets
    X_val, X_val_ens, y_val, y_val_ens = train_test_split(
        X_val, y_val, test_size=validate_ensemble_size, random_state=random_state)

    return X_train, X_test, X_val, X_val_ens, y_train, y_test, y_val, y_val_ens