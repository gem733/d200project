"""
Train a meta-learner ensemble using predictions of base models.

- Uses validation set to train the meta-model
- Evaluates on test set (MSE, R²)
- Returns ensemble predictions and fitted meta-model
- Keeps SHAP-compatible base models for analysis
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_meta_ensemble(models, x_val_ens_scaled, y_val_ens, x_test_scaled, y_test):
    """
    Train a meta-learner ensemble.

    Parameters
    ----------
    models : dict
        Dictionary of trained base models {"Post-Lasso": model, "Random Forest": model, "Neural Network": model}
    x_val_ens_scaled : np.ndarray
        Features of the validation set for ensemble weighting
    y_val_ens : pd.Series
        Target values for ensemble validation
    x_test_scaled : np.ndarray
        Features of the test set
    y_test : pd.Series
        True target values for test set

    Returns
    -------
    meta_model : sklearn.linear_model.LinearRegression
        Trained meta-model
    ensemble_pred : np.ndarray
        Predictions of the ensemble on the test set
    weights : dict
        Weight (coefficient) of each base model in the meta-model
    """

    # Make base predictions on validation ensemble set
    val_preds = []
    for name, model in models.items():
        if name == "Neural Network":
            # Neural network expects a torch tensor
            import torch
            X_val_tensor = torch.tensor(x_val_ens_scaled, dtype=torch.float32)
            val_preds.append(model.predict(X_val_tensor).numpy())
        else:
            val_preds.append(model.predict(x_val_ens_scaled))
    
    # Stack predictions as features for meta-model
    meta_X = np.column_stack(val_preds)
    meta_y = y_val_ens.values

    # Train meta-model (linear regression)
    meta_model = LinearRegression()
    meta_model.fit(meta_X, meta_y)

    # Get weights for each base model
    ensemble_weights = dict(zip(models.keys(), meta_model.coef_))

    # Make predictions on test set
    test_preds = []
    for name, model in models.items():
        if name == "Neural Network":
            X_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
            test_preds.append(model.predict(X_test_tensor).numpy())
        else:
            test_preds.append(model.predict(x_test_scaled))

    meta_X_test = np.column_stack(test_preds)
    meta_preds = meta_model.predict(meta_X_test)

    # Evaluate
    mse = mean_squared_error(y_test, meta_preds)
    r2 = r2_score(y_test, meta_preds)

    print("Meta-Learner Ensemble trained")
    print("Ensemble weights:")
    for name, w in ensemble_weights.items():
        print(f"{name}: {w:.4f}")
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

    return meta_model, meta_preds, ensemble_weights