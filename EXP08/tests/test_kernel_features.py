from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "EXP08"
sys.path.insert(0, str(EXP))

from run_experiment import KERNEL_SIZES, KernelCNN


def test_kernel_cnn_forward_shape():
    import torch

    model = KernelCNN(kernel_size=3)
    logits = model(torch.zeros(3, 1, 8, 8))
    assert logits.shape == (3, 10)


def test_kernel_variants_are_required_five_odd_sizes():
    assert KERNEL_SIZES == [1, 3, 5, 7, 9]


def test_padding_preserves_convolution_shape():
    import torch

    for kernel_size in KERNEL_SIZES:
        model = KernelCNN(kernel_size=kernel_size)
        x = torch.zeros(2, 1, 8, 8)
        conv_out = model.activation(model.conv(x))
        assert conv_out.shape == (2, 8, 8, 8)
        assert model.padding == kernel_size // 2
