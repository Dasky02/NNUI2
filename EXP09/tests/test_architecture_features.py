from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "EXP09"
sys.path.insert(0, str(EXP))

from run_experiment import ARCHITECTURES, CLASS_NAMES, FLOWER_DATASET_DIR, FlexibleCNN, prepare_data


def test_flexible_cnn_forward_shape():
    import torch

    model = FlexibleCNN(ARCHITECTURES[0])
    logits = model(torch.zeros(4, 3, 64, 64))
    assert logits.shape == (4, 5)


def test_uses_local_flower_dataset_with_five_classes():
    assert FLOWER_DATASET_DIR == ROOT.parent / "cviceni" / "dataset"
    assert FLOWER_DATASET_DIR.exists()
    assert CLASS_NAMES == ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    for class_name in CLASS_NAMES:
        assert (FLOWER_DATASET_DIR / class_name).is_dir()


def test_prepare_data_loads_flower_dataset():
    images, train_idx, val_idx, test_idx, _y_train, _y_val, _y_test, class_names, dataset_info = prepare_data()
    assert images.shape[1:] == (3, 64, 64)
    assert class_names == CLASS_NAMES
    assert dataset_info["path"] == str(FLOWER_DATASET_DIR)
    assert dataset_info["classes"] == 5
    assert dataset_info["images"] == len(train_idx) + len(val_idx) + len(test_idx)
    assert "sklearn" not in str(dataset_info["name"]).lower()


def test_five_architectures_with_structural_differences():
    assert len(ARCHITECTURES) == 5
    conv_counts = [len(config["channels"]) for config in ARCHITECTURES]
    dropouts = [float(config["dropout"]) for config in ARCHITECTURES]
    activations = {str(config["activation"]) for config in ARCHITECTURES}
    fc_sizes = {int(config["fc_hidden"]) for config in ARCHITECTURES}
    assert conv_counts == [1, 2, 3, 2, 3]
    assert max(dropouts) > 0.0
    assert {"relu", "tanh"} <= activations
    assert len(fc_sizes) > 1


def test_architecture_names_match_assignment_variants():
    assert [config["name"].split("_")[0] for config in ARCHITECTURES] == ["A1", "A2", "A3", "A4", "A5"]


def test_metrics_csv_has_required_columns_if_present():
    import csv

    metrics = EXP / "results" / "metrics.csv"
    if not metrics.exists():
        return
    with metrics.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert {
            "architecture",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "test_loss",
        } <= set(reader.fieldnames or [])


def test_reports_do_not_claim_digits_as_main_dataset():
    for path in [ROOT / "experiment_09.md", EXP / "report" / "report.md"]:
        if path.exists():
            text = path.read_text(encoding="utf-8").lower()
            assert "sklearn.datasets.load_digits" not in text
