"""
Training pipeline:
- Loads and preprocesses data
- One-hot encodes region
- Drops unwanted columns
- Normalizes features
- Trains Post-Lasso, Random Forest, Neural Network
- Evaluates models on test set
- Keeps models and scalers for SHAP and plotting
"""

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from src.data.load_data import load_data
from src.transformations.convert_strings import convert_strings
from src.transformations.one_hot_encode import one_hot_encode
from src.transformations.replace_missing import replace_missing

from src.models.post_lasso import train_post_lasso
from src.models.random_forest import train_random_forest
from src.models.neural_network import train_neural_network

from src.data.create_split import create_split


def train_models(file_path: str, epochs_nn: int = 1500):
    """
    Complete ML pipeline:
    - Loads and preprocesses data
    - Normalizes features
    - Trains post-lasso, random forest, neural network
    - Evaluates models on test set
    - Returns trained models and scalers
    """
    
    # Load data
    df = load_data(file_path)

    # Drop unwanted columns
    drop_cols = ["area", "votes", "last_party", "last_election", 
                 "swing", "year", "majority", "constituency"]
    df = df.drop(columns=drop_cols, errors='ignore')  # errors='ignore' in case some columns missing

    # Preprocess
    cols_to_convert = df.select_dtypes(include="object").columns.tolist()
    cols_to_convert = [c for c in cols_to_convert if c != "region"]
    df = convert_strings(df, cols_to_convert)

    if 'region' in df.columns:
        df = one_hot_encode(df, 'region')

    df = replace_missing(df)

    # Split data
    target_col = "margin"
    splits = create_split(df, target_col)
    X_train, X_test, X_val, X_val_ens, y_train, y_test, y_val, y_val_ens = splits

    # Normalize
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_val_scaled = scaler.transform(X_val)
    x_val_ens_scaled = scaler.transform(X_val_ens)
    x_test_scaled = scaler.transform(X_test)

    # Train Post-Lasso
    best_lasso_model, best_alpha = train_post_lasso(
        x_train_scaled, y_train.to_numpy(),
        x_val_scaled, y_val.to_numpy()
    )

    # Train Random Forest
    best_rf_model, best_rf_params = train_random_forest(
        x_train_scaled, y_train.to_numpy(),
        x_val_scaled, y_val.to_numpy()
    )

    # Train Neural Network
    X_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_t = torch.tensor(x_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    best_nn_model, best_nn_params = train_neural_network(
        X_train_t, y_train_t,
        X_val_t, y_val_t,
        epochs=epochs_nn
    )

    # Evaluate Models
    models = {
        "Post-Lasso": best_lasso_model,
        "Random Forest": best_rf_model,
        "Neural Network": best_nn_model
    }

    for name, model in models.items():
        if name == "Neural Network":
            y_pred = model.predict(X_test_t).numpy()
        else:
            y_pred = model.predict(x_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")

    # Return models and scalers for SHAP / plotting / ensemble
    return {
        "models": models,
        "scaler": scaler,
        "x_train_scaled": x_train_scaled,
        "y_train": y_train,
        "x_test_scaled": x_test_scaled,
        "y_test": y_test,
        "x_val_scaled": x_val_scaled,
        "x_val_ens_scaled": x_val_ens_scaled,
        "y_val": y_val,
        "y_val_ens": y_val_ens
}