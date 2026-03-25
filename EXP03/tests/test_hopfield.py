"""Unit tests for the discrete Hopfield network."""

from __future__ import annotations

import numpy as np

from hopfield.net import HopfieldNet


def _toy_pattern() -> np.ndarray:
    return np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=float,
    )


def test_symmetry_and_zero_diag() -> None:
    p1 = _toy_pattern()
    p2 = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    net = HopfieldNet(input_size=9, bipolar=True)
    net.train([p1, p2])

    assert np.allclose(net.W, net.W.T)
    assert np.allclose(np.diag(net.W), 0.0)


def test_energy_decreases_or_not_increase_on_recall() -> None:
    p = _toy_pattern()
    net = HopfieldNet(input_size=9, bipolar=True)
    net.train([p])

    noisy = p.reshape(-1).copy()
    noisy[[1, 3]] = 1.0  # small perturbation from original pattern

    result = net.recall(noisy, max_iters=10)
    energies = result["energies"]

    assert len(energies) >= 2
    assert energies[-1] <= energies[0] + 1e-9


def test_recall_returns_stored_pattern_from_noisy_input() -> None:
    p = _toy_pattern()
    net = HopfieldNet(input_size=9, bipolar=True)
    net.train([p])

    noisy = p.reshape(-1).copy()
    noisy[1] = 1.0 - noisy[1]

    result = net.recall(noisy, max_iters=10)
    recalled = result["final_state"]
    expected = net.preprocess(p)[0]

    assert np.array_equal(recalled, expected)
