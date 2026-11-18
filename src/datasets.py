"""Small synthetic datasets for Neural Nexus."""
from __future__ import annotations

import numpy as np


def make_dataset(
    name: str = "circles",
    n_samples: int = 400,
    seed: int = 7,
):
    """Generate shuffled raw features and binary labels.

    Standardization is deliberately left to the backend.
    The backend must split the data first and fit its scaler
    using training data only.
    """
    rng = np.random.default_rng(seed)
    name = name.lower()

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    if name == "xor":
        centers = np.array([
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1],
        ], dtype=float)

        labels = np.array([0, 1, 1, 0])
        ids = rng.integers(0, 4, size=n_samples)

        X = centers[ids] + rng.normal(
            0, 0.28, size=(n_samples, 2)
        )
        y = labels[ids].reshape(-1, 1).astype(float)

    elif name == "moons":
        half = n_samples // 2
        other_half = n_samples - half

        t1 = rng.uniform(0, np.pi, half)
        t2 = rng.uniform(0, np.pi, other_half)

        upper = np.column_stack([
            np.cos(t1),
            np.sin(t1),
        ])

        lower = np.column_stack([
            1 - np.cos(t2),
            0.5 - np.sin(t2),
        ])

        X = np.vstack([upper, lower])
        X += rng.normal(
            0, 0.09, size=(n_samples, 2)
        )

        y = np.vstack([
            np.zeros((half, 1)),
            np.ones((other_half, 1)),
        ])

    elif name == "circles":
        labels = rng.integers(0, 2, size=n_samples)
        angles = rng.uniform(0, 2 * np.pi, size=n_samples)

        radii = np.where(
            labels == 0,
            0.55,
            1.15,
        ) + rng.normal(0, 0.08, n_samples)

        X = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles),
        ])

        y = labels.reshape(-1, 1).astype(float)

    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            "Choose circles, moons, or xor."
        )

    # Shuffle without calculating statistics from the full dataset.
    order = rng.permutation(len(X))
    X = X[order]
    y = y[order]

    return X.astype(np.float64), y.astype(np.float64)
