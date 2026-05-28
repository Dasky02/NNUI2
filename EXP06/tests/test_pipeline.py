from __future__ import annotations

from run_experiment import FFNNClassifier, build_topologies
import torch


def test_topology_count_is_five() -> None:
    topologies = build_topologies()
    assert len(topologies) == 5
    assert topologies[0]["hidden_layer_sizes"] == (8,)
    assert all("optimizer" in topology for topology in topologies)
    assert all("epochs" in topology for topology in topologies)


def test_ffnn_forward_shape() -> None:
    model = FFNNClassifier(input_dim=13, hidden_layers=(8,), num_classes=3, activation="relu")
    output = model(torch.zeros((4, 13), dtype=torch.float32))
    assert output.shape == (4, 3)
