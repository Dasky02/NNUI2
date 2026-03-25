from __future__ import annotations

import numpy as np

from som import KohonenSOM


def test_neighborhood_respects_bounds() -> None:
    som = KohonenSOM(input_dim=2, n_units=3, radius=1, epochs=5)
    assert som.neighborhood(0, 1) == [1]
    assert som.neighborhood(1, 1) == [0, 2]
    assert som.neighborhood(2, 1) == [1]


def test_training_records_history_and_reduces_error() -> None:
    rng = np.random.default_rng(123)
    X = np.vstack(
        [
            rng.normal(loc=(0.0, 0.0), scale=0.3, size=(40, 2)),
            rng.normal(loc=(3.0, 0.0), scale=0.3, size=(40, 2)),
            rng.normal(loc=(0.0, 3.0), scale=0.3, size=(40, 2)),
        ]
    )
    som = KohonenSOM(input_dim=2, n_units=3, lr=0.5, radius=1, epochs=40, seed=1)
    initial_error = som.quantization_error(X)
    history = som.train(X)

    assert len(history) == 40
    assert history[-1] < initial_error
