from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "EXP07"
sys.path.insert(0, str(EXP))

from run_experiment import FeatureMLP, build_variants, hog_features
import torch


def test_hog_features_shape_and_finiteness():
    images = np.zeros((2, 8, 8), dtype=float)
    features = hog_features(images, cells=4, bins=9)
    assert features.shape == (2, 4 * 4 * 9)
    assert np.isfinite(features).all()


def test_feature_variants_are_distinct_representations():
    rng = np.random.default_rng(42)
    images = rng.random((32, 8, 8), dtype=np.float32)
    flat = images.reshape(len(images), -1)
    train_idx = np.arange(28)
    variants = build_variants(flat, images, train_idx)
    assert set(variants) == {"raw_pixels_64", "hog_4x4_9bins", "pca_16_from_raw", "hog_pca_24"}
    assert {variant.dimension for variant in variants.values()} == {16, 24, 64, 144}


def test_ffnn_forward_shape():
    model = FeatureMLP(input_dim=64, hidden_layers=(48, 24), num_classes=10)
    output = model(torch.zeros((3, 64), dtype=torch.float32))
    assert output.shape == (3, 10)
