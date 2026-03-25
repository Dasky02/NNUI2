from __future__ import annotations

from run_experiment import build_topologies


def test_topology_count_is_five() -> None:
    topologies = build_topologies()
    assert len(topologies) == 5
    assert topologies[0]["hidden_layer_sizes"] == (8,)
