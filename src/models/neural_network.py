import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    """
    Simple feedforward neural network for regression.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

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
    hidden_dims=None,
    learning_rates=None,
    epochs=100
):
    """
    Train a neural network and tune hyperparameters using validation data.

    Parameters
    ----------
    X_train : torch.Tensor
    y_train : torch.Tensor
    X_val : torch.Tensor
    y_val : torch.Tensor
    hidden_dims : list
        Hidden layer sizes to search over
    learning_rates : list
        Learning rates to search over
    epochs : int
        Number of training epochs

    Returns
    -------
    best_model : torch.nn.Module
    best_params : dict
    """

    if hidden_dims is None:
        hidden_dims = [16, 32, 64]

    if learning_rates is None:
        learning_rates = [0.001, 0.01]

    input_dim = X_train.shape[1]
    loss_fn = nn.MSELoss()

    best_mse = float("inf")
    best_nn_params = None

    # Hyperparameter tuning
    for hidden_dim in hidden_dims:
        for lr in learning_rates:

            model = NeuralNetwork(input_dim, hidden_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}")

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
                    "hidden_dim": hidden_dim,
                    "learning_rate": lr
                }

    # Train final model on TRAIN + VALIDATION
    X_combined = torch.cat([X_train, X_val])
    y_combined = torch.cat([y_train, y_val])

    best_nn_model = NeuralNetwork(input_dim, best_nn_params["hidden_dim"])
    optimizer = optim.Adam(best_nn_model.parameters(), lr=best_nn_params["learning_rate"])

    for epoch in range(epochs):

        best_nn_model.train()

        optimizer.zero_grad()

        predictions = best_nn_model(X_combined).squeeze()
        loss = loss_fn(predictions, y_combined)

        loss.backward()
        optimizer.step()

    return best_nn_model, best_nn_params