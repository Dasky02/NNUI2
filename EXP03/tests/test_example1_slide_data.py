"""Hopfield test using slide data from Example 1."""

from __future__ import annotations

from typing import Any

import numpy as np

from hopfield.net import HopfieldNet


def _extract_recall_parts(result: Any) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Support both dict and tuple/list return conventions for recall()."""

    if isinstance(result, dict):
        final_state = np.asarray(result["final_state"], dtype=float)
        states = [np.asarray(s, dtype=float) for s in result.get("states", [])]
        info = dict(result.get("info", {}))
        return final_state, states, info

    if isinstance(result, (tuple, list)):
        if len(result) < 2:
            raise AssertionError("recall() tuple/list result must contain at least final_state and states")
        final_state = np.asarray(result[0], dtype=float)
        states = [np.asarray(s, dtype=float) for s in result[1]]
        info = dict(result[3]) if len(result) >= 4 and isinstance(result[3], dict) else {}
        return final_state, states, info

    raise AssertionError(f"Unsupported recall() result type: {type(result)!r}")


def test_example1_slide_weights_and_recall_trace() -> None:
    x1 = np.array([1, 1, 1, -1], dtype=float)
    x2 = np.array([1, 1, -1, -1], dtype=float)

    net = HopfieldNet(input_size=4, bipolar=True)
    net.train([x1, x2])

    W_expected = np.array(
        [
            [0, 2, 0, -2],
            [2, 0, 0, -2],
            [0, 0, 0, 0],
            [-2, -2, 0, 0],
        ],
        dtype=float,
    )
    assert np.array_equal(net.W, W_expected)

    s0 = np.array([-1, 1, -1, -1], dtype=float)
    result = net.recall(s0, max_iters=10, stop_when_stable=True, detect_2cycle=True)
    final_state, states, info = _extract_recall_parts(result)

    # If an implementation omits the initial state from history, prepend it to
    # keep the expected indices aligned with the slide notation.
    if not states or not np.array_equal(states[0], s0):
        states = [s0.copy(), *states]

    assert len(states) >= 3
    assert np.array_equal(states[0], s0)
    assert np.array_equal(states[1], np.array([1, 1, 1, 1], dtype=float))
    assert np.array_equal(states[2], np.array([1, 1, 1, -1], dtype=float))

    assert np.array_equal(final_state, x1)

    if info:
        assert info.get("converged", False) is True or np.array_equal(final_state, x1)
        if "reason" in info:
            assert info["reason"] in {"fixed_point", "max_iters"}
