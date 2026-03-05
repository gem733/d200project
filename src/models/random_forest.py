from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_random_forest(X_train, y_train, X_val, y_val,
                       n_estimators_list=None, max_depth_list=None, random_state=42):
    """
    Trains a Random Forest model by selecting the best hyperparameters
    using a validation set.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation targets
        n_estimators_list (list): List of tree counts to search over
        max_depth_list (list): List of max tree depths to search over
        random_state (int): Random seed

    Returns:
        best_rf_model: Trained RF model
        best_rf_params: Selected hyperparameters
    """

    if n_estimators_list is None:
        n_estimators_list = [100, 200, 500]

    if max_depth_list is None:
        max_depth_list = [None, 5, 10, 12, 20]

    best_mse = float("inf")
    best_rf_params = None

    # Grid search over hyperparameters
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse = mean_squared_error(y_val, y_pred)

            if mse < best_mse:
                best_mse = mse
                best_rf_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

    # Train final model on combined train + validation data
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    best_rf_model = RandomForestRegressor(
        n_estimators=best_rf_params["n_estimators"],
        max_depth=best_rf_params["max_depth"],
        random_state=random_state,
        n_jobs=-1
    )

    best_rf_model.fit(X_combined, y_combined)

    return best_rf_model, best_rf_params