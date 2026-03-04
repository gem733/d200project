from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

def train_post_lasso(X_train, y_train, X_val, y_val, alphas=None):
    """
    Trains a post-Lasso model by selecting the best alpha using a validation set.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation targets
        alphas: List of alpha values to search over.
            Defaults to 20 log-spaced values between 0.001 and 10.

    Returns:
        best_model (sklearn.linear_model.Lasso): Trained Lasso model with best alpha
        best_alpha (float): Selected alpha value
    """

    if alphas is None:
        alphas = np.logspace(-3, 1, 20)

    best_alpha = None
    best_mse = float("inf")

    # Grid search over alpha
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    # Train final model on combined train + validation data
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    best_lasso_model = Lasso(alpha=best_alpha, max_iter=10000)
    best_lasso_model.fit(X_combined, y_combined)

    return best_lasso_model, best_alpha