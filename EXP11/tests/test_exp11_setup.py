from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "EXP11"
sys.path.insert(0, str(EXP))

from run_experiment import (
    CLASS_NAMES,
    POTATOES_DATASET_DIR,
    SEGMENTACE_TXT,
    TinySegNet,
    find_samples,
    inspect_dataset,
    parse_yolo_polygons,
)


def test_uses_local_potatoes_dataset():
    assert POTATOES_DATASET_DIR == ROOT.parent / "cviceni" / "Potatoes_seg"
    assert POTATOES_DATASET_DIR.exists()
    assert (POTATOES_DATASET_DIR / "data.yaml").exists()
    assert SEGMENTACE_TXT == ROOT.parent / "cviceni" / "segmentace.txt"
    assert SEGMENTACE_TXT.exists()


def test_potatoes_dataset_has_segmentation_structure():
    info = inspect_dataset()
    assert info["status"] == "OK"
    assert info["has_images"] is True
    assert info["has_masks_or_annotations"] is True
    assert info["nc"] == 1
    assert info["names"] == CLASS_NAMES
    assert info["splits"]["train"]["paired_samples"] > 0
    assert info["splits"]["valid"]["paired_samples"] > 0
    assert info["splits"]["test"]["paired_samples"] > 0


def test_yolo_polygon_labels_are_parseable():
    sample = find_samples("train")[0]
    polygons = parse_yolo_polygons(sample.label_path, width=100, height=100)
    assert polygons
    assert all(len(polygon) >= 3 for polygon in polygons)


def test_tiny_segnet_forward_shape():
    import torch

    model = TinySegNet()
    logits = model(torch.zeros(2, 3, 128, 128))
    assert logits.shape == (2, 1, 128, 128)


def test_metrics_csv_has_required_columns_if_present():
    import csv

    metrics = EXP / "results" / "metrics.csv"
    if not metrics.exists():
        return
    with metrics.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert {
            "epoch",
            "train_loss",
            "val_loss",
            "val_pixel_accuracy",
            "val_iou",
            "val_dice",
            "test_loss",
            "test_pixel_accuracy",
            "test_iou",
            "test_dice",
        } <= set(reader.fieldnames or [])


def test_report_does_not_use_flower_classification_dataset():
    for path in [ROOT / "experiment_11.md", EXP / "report" / "report.md"]:
        if path.exists():
            text = path.read_text(encoding="utf-8").lower()
            assert "/cviceni/dataset/" not in text
            assert "pycharmprojects/cviceni/dataset" not in text
