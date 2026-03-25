from __future__ import annotations

import numpy as np

from ffnn import FFNN


def test_forward_shapes_match_configuration() -> None:
    net = FFNN(input_dim=4, hidden_units=5, output_dim=1, f_hidden="tanh", f_output="linear", lr=0.01)
    y, h, h_b, x_b = net.forward(np.array([0.1, 0.2, 0.3, 0.4], dtype=float))
    assert y.shape == (1,)
    assert h.shape == (5,)
    assert h_b.shape == (6,)
    assert x_b.shape == (5,)


def test_training_reduces_error_on_simple_problem() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(80, 2))
    y = (X[:, [0]] * 0.7 + X[:, [1]] * -0.2 + 0.1)
    net = FFNN(input_dim=2, hidden_units=6, output_dim=1, f_hidden="tanh", f_output="linear", lr=0.05, seed=1)
    initial_error = net.validate(X, y)
    train_errors, _ = net.train(X, y, epochs=80, lr=0.05)
    assert train_errors[-1] < initial_error
