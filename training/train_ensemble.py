import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize


def ensemble_models(models, x_val_ens_scaled, y_val_ens, x_test_scaled, y_test):
    """
    Create a weighted ensemble of models.
    
    Parameters:
    - models (dict): Dictionary of trained models with keys: 'Post-Lasso', 'Random Forest', 'Neural Network'
    - x_val_ens_scaled: Features for ensemble validation (numpy array or torch tensor for NN)
    - y_val_ens: Target values for ensemble validation
    - x_test_scaled: Test set features
    - y_test: Test set targets
    
    Returns:
    - ensemble_preds_test: Ensemble predictions on test set
    - weights: Optimized weights for each model
    - mse: MSE on test set
    - r2: R² on test set
    """
    # Make predictions on ensemble validation set
    val_preds = []
    for name, model in models.items():
        if name == "Neural Network":
            import torch
            x_val_tensor = torch.tensor(x_val_ens_scaled, dtype=torch.float32)
            preds = model.predict(x_val_tensor).detach().numpy().squeeze()
        else:
            preds = model.predict(x_val_ens_scaled)
        val_preds.append(preds)
    
    val_preds = np.array(val_preds)
    

    # Fit weights

    def loss_fn(w):
        ensemble_pred = np.dot(w, val_preds)
        return mean_squared_error(y_val_ens, ensemble_pred)
    
    # Initial guess:
    n_models = val_preds.shape[0]
    w0 = np.ones(n_models) / n_models
    
    # Constraints:
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum = 1
    )
    bounds = [(0, 1) for _ in range(n_models)]
    
    res = minimize(loss_fn, w0, bounds=bounds, constraints=constraints)
    weights = res.x


    # Make predictions on test set

    test_preds = []
    for i, (name, model) in enumerate(models.items()):
        if name == "Neural Network":
            import torch
            x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
            preds = model.predict(x_test_tensor).detach().numpy().squeeze()
        else:
            preds = model.predict(x_test_scaled)
        test_preds.append(preds)
    
    test_preds = np.array(test_preds)  # shape: (n_models, n_samples)
    ensemble_preds_test = np.dot(weights, test_preds)
    
    # Evaluate
    mse = mean_squared_error(y_test, ensemble_preds_test)
    r2 = r2_score(y_test, ensemble_preds_test)
    
    # Return results
    print("Ensemble weights:")
    for w, name in zip(weights, models.keys()):
        print(f"{name}: {w:.3f}")
    
    print(f"Ensemble MSE (test): {mse:.4f}")
    print(f"Ensemble R² (test): {r2:.4f}")
    
    return ensemble_preds_test, weights, mse, r2