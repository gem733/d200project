"""
Pipeline:
- Runs the full ML pipeline using train_models
- Loads, preprocesses, normalizes data
- Trains Post-Lasso, Random Forest, Neural Network
- Prints evaluation metrics
- Returns models and scalers for further analysis (SHAP, plotting, ensemble)
"""

import os
from pathlib import Path

# Ensure the src and training folders are in your path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds root folder

from training.train_ensemble import ensemble_models
from training.train_models import train_models
from sklearn.metrics import mean_squared_error, r2_score
import torch

def main():
    # Path to raw CSV file
    file_path = os.path.join("data", "raw", "constituencies_dataset.csv")
    
    # Run the full pipeline
    results = train_models(file_path)
    
    models = results["models"]
    scaler = results["scaler"]
    
    # extracted the preprocessed data for the ensemble method

    x_train_scaled = results["x_train_scaled"]
    x_val_scaled = results["x_val_scaled"]
    x_val_ens_scaled = results["x_val_ens_scaled"]
    x_test_scaled = results["x_test_scaled"]
    
    y_train = results["y_train"]
    y_val = results["y_val"]
    y_val_ens = results["y_val_ens"]
    y_test = results["y_test"]
    
    
    # Train ensemble
    ensemble_preds, ensemble_weights, ensemble_mse, ensemble_r2 = ensemble_models(
        models,
        x_val_ens_scaled,
        y_val_ens,
        x_test_scaled,
        y_test
    )
    
    
    print("\nIndividual model test metrics:")
    for name, model in models.items():
        if name == "Neural Network":
            x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
            y_pred = model.predict(x_test_tensor).detach().numpy().squeeze()
        else:
            y_pred = model.predict(x_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    

    # Return everything for further analysis

    return {
        "models": models,
        "scaler": scaler,
        "x_test_scaled": x_test_scaled,
        "y_test": y_test,
        "ensemble_preds": ensemble_preds,
        "ensemble_weights": ensemble_weights
    }

if __name__ == "__main__":
    main()