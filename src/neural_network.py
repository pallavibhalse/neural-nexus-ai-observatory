
"""A tiny multilayer perceptron implemented with NumPy and manual backpropagation."""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class NeuralNetwork:
    """Binary classifier: 2 inputs -> one ReLU hidden layer -> sigmoid output."""

    def __init__(self, input_size=2, hidden_size=8, learning_rate=0.03, seed=42):
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)

        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(
            0, np.sqrt(2 / input_size), (input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(
            0, np.sqrt(1 / hidden_size), (hidden_size, 1)
        )
        self.b2 = np.zeros((1, 1))

        self.history = []

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = _sigmoid(Z2)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "A2": A2,
        }

        return A2, cache

    @staticmethod
    def binary_cross_entropy(y, probabilities):
        eps = 1e-8
        p = np.clip(probabilities, eps, 1 - eps)

        return float(
            -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        )

    def _evaluate(self, X, y):
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= 0.5).astype(float)

        return {
            "loss": self.binary_cross_entropy(y, probabilities),
            "accuracy": float(np.mean(predictions == y)),
        }
    def compute_gradients(self, X, y):
        probs, cache = self.forward(X)
        n = X.shape[0]

        dZ2 = probs - y
        dW2 = cache["A1"].T @ dZ2 / n
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)

        dW1 = cache["X"].T @ dZ1 / n
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

    def train(
        self,
        X,
        y,
        epochs=300,
        record_every=5,
        X_val=None,
        y_val=None,
    ):
        """Train on training data and optionally evaluate validation data."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError("X must have shape (samples, input_size)")

        if y.shape != (X.shape[0], 1):
            raise ValueError("y must have shape (samples, 1)")

        if (X_val is None) != (y_val is None):
            raise ValueError("Provide both X_val and y_val, or neither")

        if X_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)

            if X_val.ndim != 2 or X_val.shape[1] != self.input_size:
                raise ValueError("X_val must have shape (samples, input_size)")

            if y_val.shape != (X_val.shape[0], 1):
                raise ValueError("y_val must have shape (samples, 1)")

        epochs = max(1, min(int(epochs), 5000))
        record_every = max(1, int(record_every))

        self.history = []
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            gradients = self.compute_gradients(X, y)

            dW1 = gradients["W1"]
            db1 = gradients["b1"]
            dW2 = gradients["W2"]
            db2 = gradients["b2"]

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

            if epoch == 1 or epoch % record_every == 0 or epoch == epochs:
                # Evaluate after updating weights so both metrics
                # represent the same model state.
                train_metrics = self._evaluate(X, y)

                record = {
                    "epoch": epoch,
                    "loss": round(train_metrics["loss"], 6),
                    "accuracy": round(train_metrics["accuracy"], 6),
                }

                if X_val is not None:
                    val_metrics = self._evaluate(X_val, y_val)
                    record["test_loss"] = round(val_metrics["loss"], 6)
                    record["test_accuracy"] = round(val_metrics["accuracy"], 6)

                self.history.append(record)

        final_train = self._evaluate(X, y)

        self.final_loss = final_train["loss"]
        self.final_accuracy = final_train["accuracy"]

        if X_val is not None:
            final_test = self._evaluate(X_val, y_val)
            self.final_test_loss = final_test["loss"]
            self.final_test_accuracy = final_test["accuracy"]

        return self.history

    def predict_proba(self, X):
        probabilities, _ = self.forward(np.asarray(X, dtype=float))
        return probabilities

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def inspect(self, X):
        probabilities, cache = self.forward(X)

        return {
            "input": X[:1].round(4).tolist(),
            "hidden_activations": cache["A1"][:1].round(4).ravel().tolist(),
            "output_probability": float(probabilities[0, 0]),
            "prediction": int(probabilities[0, 0] >= 0.5),
            "weights_input_hidden": self.W1.round(4).tolist(),
            "weights_hidden_output": self.W2.round(4).ravel().tolist(),
        }
