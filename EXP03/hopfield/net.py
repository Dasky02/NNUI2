"""Discrete Hopfield network implementation."""

from __future__ import annotations

from typing import Any

import numpy as np


class HopfieldNet:
    """Discrete Hopfield network with synchronous recall updates.

    Notes:
        The weight matrix is trained using the Hebbian rule:
        ``W = sum(p p^T)`` with zero diagonal.
    """

    def __init__(self, input_size: int, bipolar: bool = True) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.input_size = int(input_size)
        self.bipolar = bool(bipolar)
        self.W = np.zeros((self.input_size, self.input_size), dtype=float)

    def preprocess(self, patterns: np.ndarray | list[Any]) -> np.ndarray:
        """Convert input patterns to a 2D array of shape ``(k, N)``.

        Accepts shapes ``(k, H, W)``, ``(k, N)``, ``(H, W)``, or ``(N,)``.
        Output dtype is ``float``.
        """

        arr = np.asarray(patterns, dtype=float)
        if arr.ndim == 0:
            raise ValueError("patterns must not be scalar")

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[1] == self.input_size:
                arr = arr.reshape(arr.shape[0], self.input_size)
            elif arr.size == self.input_size:
                arr = arr.reshape(1, self.input_size)
            else:
                raise ValueError(
                    f"2D patterns with shape {arr.shape} do not match input_size={self.input_size}"
                )
        elif arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        else:
            raise ValueError("patterns must have 1, 2, or 3 dimensions")

        if arr.shape[1] != self.input_size:
            raise ValueError(
                f"Flattened pattern size {arr.shape[1]} does not match input_size={self.input_size}"
            )

        arr = self._normalize_state_values(arr)
        return arr.astype(float, copy=False)

    def train(self, patterns: np.ndarray | list[Any]) -> None:
        """Train Hopfield weights using Hebbian learning from given patterns.

        Uses the unnormalized Hebbian rule ``W = sum(p p^T)`` and then enforces
        zero diagonal and symmetry.
        """

        P = self.preprocess(patterns)
        if P.size == 0:
            raise ValueError("No patterns provided for training")

        W = np.zeros_like(self.W)
        for p in P:
            W += np.outer(p, p)

        np.fill_diagonal(W, 0.0)
        W = 0.5 * (W + W.T)
        self.W = W

    def energy(self, state: np.ndarray | list[Any]) -> float:
        """Compute Hopfield energy ``E = -0.5 * s^T W s`` for a single state."""

        s = self.preprocess(state)[0]
        return float(-0.5 * s.T @ self.W @ s)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Activation for synchronous updates."""

        if self.bipolar:
            return np.where(x >= 0.0, 1.0, -1.0)
        return np.where(x >= 0.5, 1.0, 0.0)

    def recall(
        self,
        initial_state: np.ndarray | list[Any],
        max_iters: int = 50,
        stop_when_stable: bool = True,
        detect_2cycle: bool = True,
    ) -> dict[str, Any]:
        """Run synchronous recall and return trajectory + metadata.

        Returns:
            dict with keys:
                ``final_state``: ndarray ``(N,)``
                ``states``: list[ndarray] including initial state
                ``energies``: list[float] aligned with ``states``
                ``info``: dict with ``converged``, ``reason``, ``iters``
        """

        if max_iters < 0:
            raise ValueError("max_iters must be >= 0")

        s = self.preprocess(initial_state)[0]
        states: list[np.ndarray] = [s.copy()]
        energies: list[float] = [self.energy(s)]

        converged = False
        reason = "max_iters"

        for _ in range(max_iters):
            s_next = self._activation(self.W @ states[-1])
            states.append(s_next.astype(float, copy=False))
            energies.append(self.energy(s_next))

            if stop_when_stable and np.array_equal(states[-1], states[-2]):
                converged = True
                reason = "fixed_point"
                break

            if detect_2cycle and len(states) >= 3 and np.array_equal(states[-1], states[-3]):
                converged = False
                reason = "2_cycle"
                break

        info = {"converged": converged, "reason": reason, "iters": len(states) - 1}
        return {
            "final_state": states[-1].copy(),
            "states": [st.copy() for st in states],
            "energies": [float(e) for e in energies],
            "info": info,
        }

    def _normalize_state_values(self, arr: np.ndarray) -> np.ndarray:
        """Convert values to bipolar or binary representation."""

        if self.bipolar:
            if self._all_values_in(arr, {0.0, 1.0}):
                return arr * 2.0 - 1.0
            if self._all_values_in(arr, {-1.0, 1.0}):
                return arr
            if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
                return np.where(arr >= 0.5, 1.0, -1.0)
            return np.where(arr >= 0.0, 1.0, -1.0)

        if self._all_values_in(arr, {-1.0, 1.0}):
            return 0.5 * (arr + 1.0)
        if self._all_values_in(arr, {0.0, 1.0}):
            return arr
        if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
            return np.where(arr >= 0.5, 1.0, 0.0)
        return np.where(arr > 0.0, 1.0, 0.0)

    @staticmethod
    def _all_values_in(arr: np.ndarray, allowed: set[float]) -> bool:
        if arr.size == 0:
            return False
        unique = np.unique(arr)
        return all(any(np.isclose(v, a) for a in allowed) for v in unique)
