from __future__ import annotations

import numpy as np


class KohonenSOM:
    def __init__(
        self,
        input_dim: int,
        n_units: int,
        lr: float = 0.5,
        radius: int = 1,
        epochs: int = 100,
        seed: int = 0,
    ) -> None:
        if input_dim <= 0 or n_units <= 0:
            raise ValueError("input_dim and n_units must be positive")
        self.input_dim = input_dim
        self.n_units = n_units
        self.lr = lr
        self.radius = radius
        self.epochs = epochs
        self.history: list[float] = []
        self._rng = np.random.default_rng(seed)
        self.W = self._rng.normal(loc=0.0, scale=1.0, size=(n_units, input_dim))

    def bmu(self, x: np.ndarray) -> int:
        distances = np.linalg.norm(self.W - x, axis=1)
        return int(np.argmin(distances))

    def neighborhood(self, k: int, radius: int) -> list[int]:
        return [idx for idx in range(max(0, k - radius), min(self.n_units, k + radius + 1)) if idx != k]

    def quantization_error(self, X: np.ndarray) -> float:
        errors = [np.linalg.norm(x - self.W[self.bmu(x)]) for x in X]
        return float(np.mean(errors))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.bmu(x) for x in X], dtype=int)

    def train(self, X: np.ndarray) -> list[float]:
        X = np.asarray(X, dtype=float)
        self.history = []
        for epoch in range(self.epochs):
            lr_t = self.lr * np.exp(-epoch / max(self.epochs, 1))
            radius_t = max(0, int(round(self.radius * np.exp(-epoch / max(self.epochs, 1)))))
            shuffled = self._rng.permutation(X)

            for x in shuffled:
                winner = self.bmu(x)
                self.W[winner] += lr_t * (x - self.W[winner])
                for neighbor in self.neighborhood(winner, radius_t):
                    distance = abs(winner - neighbor)
                    influence = lr_t / max(distance + 1, 1)
                    self.W[neighbor] += influence * (x - self.W[neighbor])

            self.history.append(self.quantization_error(X))
        return self.history

