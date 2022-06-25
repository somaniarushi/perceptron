import numpy as np
from typing import Dict

class Perceptron:
    def __init__(self, hyperparameters: Dict[str, int]):
        """
        Accepts a dictionary of hyperparameters and initializes a perceptron.
        @param hyperparameters: A dictionary of hyperparameters.
        """
        # Hyperparameters
        self.learning_rate = hyperparameters['learning_rate']
        self.epochs = hyperparameters['epochs']

        # Parameters
        self.weights = None
        self.bias = None

    def activation(self, x: np.ndarray) -> np.ndarray:
        """
        @param x: A numpy array of shape (n, 1)
        @return: 1 where x > 0, 0 otherwise (unit step function)
        """
        return np.where(x > 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits this instance of perceptron to the training data X and y.
        """
        # Initialize weights and bias
        self.weights = np.random.rand(X.shape[1], 1)
        self.bias = np.random.rand(1)

        # Why initialize weights and bias with random values? Because if we initialize with 0,
        # the weights will remain 0 and the network will not learn.

        # Train the network
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Forward pass
                z = np.dot(x_i, self.weights) + self.bias
                a = self.activation(z)

                # Backpropagation
                error = (y[idx] - a)[0]
                update = self.learning_rate * error

                # Update weights and bias
                self.weights += update * x_i
                self.bias += self.learning_rate * error

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        @param X: A numpy array of shape (n, 1)
        @return: A numpy array of shape (n, 1) containing the predictions.
        """
        return self.activation(np.dot(X, self.weights) + self.bias)