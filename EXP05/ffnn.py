from __future__ import annotations

import numpy as np


class FFNN:
    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        output_dim: int,
        f_hidden: str,
        f_output: str,
        lr: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.f_hidden = f_hidden
        self.f_output = f_output
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.V = rng.uniform(-0.5, 0.5, size=(hidden_units, input_dim + 1))
        self.W = rng.uniform(-0.5, 0.5, size=(output_dim, hidden_units + 1))

    def normalize(
        self,
        X: np.ndarray,
        T: np.ndarray,
        int_min: float,
        int_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        T_min = T.min(axis=0)
        T_max = T.max(axis=0)

        X_scale = np.where(X_max - X_min == 0.0, 1.0, X_max - X_min)
        T_scale = np.where(T_max - T_min == 0.0, 1.0, T_max - T_min)

        Xn = int_min + (X - X_min) * (int_max - int_min) / X_scale
        Tn = int_min + (T - T_min) * (int_max - int_min) / T_scale
        return Xn, Tn

    def activation(self, x: np.ndarray, function_name: str) -> np.ndarray:
        if function_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
        if function_name == "relu":
            return np.maximum(0.0, x)
        if function_name == "tanh":
            return np.tanh(x)
        if function_name == "linear":
            return x
        raise ValueError(f"Unsupported activation: {function_name}")

    def activation_derivative(self, y: np.ndarray, function_name: str) -> np.ndarray:
        if function_name == "sigmoid":
            return y * (1.0 - y)
        if function_name == "relu":
            return (y > 0.0).astype(float)
        if function_name == "tanh":
            return 1.0 - y**2
        if function_name == "linear":
            return np.ones_like(y)
        raise ValueError(f"Unsupported activation: {function_name}")

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_b = np.append(x, 1.0)
        net_h = self.V @ x_b
        h = self.activation(net_h, self.f_hidden)
        h_b = np.append(h, 1.0)
        net_y = self.W @ h_b
        y = self.activation(net_y, self.f_output)
        return y, h, h_b, x_b

    def train_epoch(self, X_train: np.ndarray, T_train: np.ndarray, lr: float | None = None) -> float:
        learning_rate = self.lr if lr is None else lr
        error = 0.0
        for x, t in zip(X_train, T_train):
            y, h, h_b, x_b = self.forward(x)
            diff = t - y
            error += float(np.sum(diff**2))

            delta_y = diff * self.activation_derivative(y, self.f_output)
            delta_h = (self.W[:, :-1].T @ delta_y) * self.activation_derivative(h, self.f_hidden)

            self.W += learning_rate * np.outer(delta_y, h_b)
            self.V += learning_rate * np.outer(delta_h, x_b)
        return error / len(X_train)

    def validate(self, X_val: np.ndarray, T_val: np.ndarray) -> float:
        errors = [np.sum((t - self.forward(x)[0]) ** 2) for x, t in zip(X_val, T_val)]
        return float(np.mean(errors))

    def test(self, X_test: np.ndarray, T_test: np.ndarray) -> float:
        return self.validate(X_test, T_test)

    def train(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        X_val: np.ndarray | None = None,
        T_val: np.ndarray | None = None,
        epochs: int = 100,
        lr: float | None = None,
    ) -> tuple[list[float], list[float]]:
        train_errors: list[float] = []
        val_errors: list[float] = []

        for _ in range(epochs):
            train_error = self.train_epoch(X_train, T_train, lr=lr)
            train_errors.append(train_error)

            if X_val is not None and T_val is not None:
                val_errors.append(self.validate(X_val, T_val))

            if not np.isfinite(train_error):
                break
        return train_errors, val_errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return self.forward(X)[0]
        return np.array([self.forward(x)[0] for x in X], dtype=float)

    def save(self, path: str) -> None:
        np.savez(
            path,
            V=self.V,
            W=self.W,
            input_dim=self.input_dim,
            hidden_units=self.hidden_units,
            output_dim=self.output_dim,
            f_hidden=self.f_hidden,
            f_output=self.f_output,
            lr=self.lr,
        )

