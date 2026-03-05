import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class NeuralNetwork(nn.Module):
    """
    Flexible feedforward neural network for regression.
    Allows multiple hidden layers.
    """

    def __init__(self, input_dim, architecture):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers dynamically
        for hidden_dim in architecture:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.ReLU())  # keeps predictions >= 0

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def train_neural_network(
    X_train,
    y_train,
    X_val,
    y_val,
    architectures=None,
    learning_rates=None,
    epochs=100
):
    """
    Train a neural network and tune hyperparameters using validation data.
    """

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Architectures to search (each list = one architecture)
    if architectures is None:
        architectures = [
            [16],
            [32],
            [64],
            [32, 16],
            [64, 32],
            [64, 32, 16]
        ]

    if learning_rates is None:
        learning_rates = [0.001, 0.01]

    input_dim = X_train.shape[1]
    loss_fn = nn.MSELoss()

    best_mse = float("inf")
    best_nn_params = None

    # Hyperparameter tuning
    for arch in architectures:
        for lr in learning_rates:

            model = NeuralNetwork(input_dim, arch)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):

                model.train()

                optimizer.zero_grad()

                predictions = model(X_train).squeeze()
                loss = loss_fn(predictions, y_train)

                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():

                val_predictions = model(X_val).squeeze()
                val_loss = loss_fn(val_predictions, y_val).item()

            if val_loss < best_mse:
                best_mse = val_loss
                best_nn_params = {
                    "architecture": arch,
                    "learning_rate": lr
                }

    # Train final model on TRAIN + VALIDATION
    X_combined = torch.cat([X_train, X_val])
    y_combined = torch.cat([y_train, y_val])

    best_nn_model = NeuralNetwork(input_dim, best_nn_params["architecture"])
    optimizer = optim.Adam(
        best_nn_model.parameters(),
        lr=best_nn_params["learning_rate"]
    )

    for epoch in range(epochs):

        best_nn_model.train()

        optimizer.zero_grad()

        predictions = best_nn_model(X_combined).squeeze()
        loss = loss_fn(predictions, y_combined)

        loss.backward()
        optimizer.step()

    print("Neural Network trained")
    print(f"Best NN Params: {best_nn_params}")

    return best_nn_model, best_nn_params