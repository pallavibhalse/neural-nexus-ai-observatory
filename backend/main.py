
"""FastAPI service for the Neural Nexus visualization frontend."""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.datasets import make_dataset
from src.neural_network import NeuralNetwork

app = FastAPI(title="Neural Nexus API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset: str = Field(default="circles", pattern="^(circles|moons|xor)$")
    epochs: int = Field(default=300, ge=1, le=5000)
    hidden_size: int = Field(default=8, ge=2, le=24)
    learning_rate: float = Field(default=0.03, gt=0, le=1)
    seed: int = Field(default=7, ge=0, le=100000)


model = None
current_data = None
current_labels = None
current_test_data = None
current_test_labels = None
current_dataset = "circles"
current_config = {
    "epochs": 300,
    "hidden_size": 8,
    "learning_rate": 0.03,
    "test_size": 0.2,
    "seed": 7,
}
current_history = []


def _metrics(X, y):
    """Calculate loss and accuracy for a supplied dataset."""
    probabilities = model.predict_proba(X)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "loss": float(model.binary_cross_entropy(y, probabilities)),
        "accuracy": float(np.mean(predictions == y)),
        "samples": int(len(X)),
    }


def _train(request: TrainRequest):
    global model
    global current_data, current_labels
    global current_test_data, current_test_labels
    global current_dataset, current_config, current_history

    # Generate a reproducible dataset.
    X, y = make_dataset(
        request.dataset,
        n_samples=400,
        seed=request.seed,
    )

    # Split indices before preprocessing to keep the test set unseen.
    rng = np.random.default_rng(request.seed)
    indices = rng.permutation(len(X))
    split_index = int(len(X) * 0.8)

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train_raw = X[train_indices]
    y_train = y[train_indices]
    X_test_raw = X[test_indices]
    y_test = y[test_indices]

    # Fit standardization using training data only.
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    # Train a fresh model on training samples only.
    model = NeuralNetwork(
        hidden_size=request.hidden_size,
        learning_rate=request.learning_rate,
        seed=request.seed + 1,
    )

    history = model.train(
    X_train,
    y_train,
    epochs=request.epochs,
    X_val=X_test,
    y_val=y_test,
)

    # Save the data and fitted preprocessing for consistent predictions.
    current_data = X_train
    current_labels = y_train
    current_test_data = X_test
    current_test_labels = y_test
    current_dataset = request.dataset
    current_history = history

    current_config = {
        "epochs": request.epochs,
        "hidden_size": request.hidden_size,
        "learning_rate": request.learning_rate,
        "test_size": 0.2,
        "seed": request.seed,
    }

    return _state(history=history)


def _state(history=None):
    if model is None:
        return {"trained": False}

    X = current_data
    y = current_labels

    train_metrics = _metrics(X, y)
    test_metrics = _metrics(current_test_data, current_test_labels)

    # Preserve the existing metrics fields used by the frontend.
    metrics = {
        **train_metrics,
        "epoch": int(current_config["epochs"]),
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_samples": test_metrics["samples"],
    }

    probs = model.predict_proba(X)
    sample = model.inspect(X)

    points = [
        {
            "x": float(X[i, 0]),
            "y": float(X[i, 1]),
            "label": int(y[i, 0]),
            "probability": float(probs[i, 0]),
        }
        for i in range(len(X))
    ]

    # Compute the decision surface in standardized feature space.
    axis = np.linspace(-2.5, 2.5, 45)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid_probs = model.predict_proba(grid).ravel()

    boundary = [
        {
            "x": float(grid[i, 0]),
            "y": float(grid[i, 1]),
            "p": float(grid_probs[i]),
        }
        for i in range(len(grid))
    ]

    return {
        "trained": True,
        "dataset": current_dataset,
        "config": current_config,
        "metrics": metrics,
        "test_metrics": test_metrics,
        "history": history if history is not None else current_history,
        "points": points,
        "boundary": boundary,
        "network": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "output_size": 1,
            "sample": sample,
            "input_hidden_weights": model.W1.round(4).tolist(),
            "hidden_output_weights": model.W2.round(4).ravel().tolist(),
        },
    }


@app.get("/api/health")
def health():
    return {"status": "online", "service": "Neural Nexus API"}


@app.get("/api/state")
def state():
    return _state()


@app.post("/api/train")
def train(request: TrainRequest):
    try:
        return _train(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
