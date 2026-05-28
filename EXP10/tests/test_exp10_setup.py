from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "EXP10"
sys.path.insert(0, str(EXP))

from run_experiment import DATASET_DOI, METRIC_FIELDS, VARIANTS, empty_metric_row, inspect_obb_labels


def test_five_yolo26_obb_variants_are_defined():
    assert len(VARIANTS) == 5
    assert [variant["name"] for variant in VARIANTS] == [
        "V1_yolo26n_obb_img80",
        "V2_yolo26n_obb_img240",
        "V3_yolo26s_obb_img240",
        "V4_yolo26s_obb_img320",
        "V5_yolo26n_obb_img320_lr005_aug",
    ]
    assert all(str(variant["requested_model"]).endswith("-obb.pt") for variant in VARIANTS)
    assert DATASET_DOI == "10.5281/zenodo.18952529"


def test_blocked_metric_row_has_required_fields(tmp_path):
    row = empty_metric_row(VARIANTS[0], "not_run", "missing dataset", None, tmp_path, "cpu", 1)
    assert list(row.keys()) == METRIC_FIELDS
    assert row["status"] == "not_run"
    assert row["map50_95"] == ""
    assert row["epochs"] == 1


def test_obb_label_checker_requires_nine_values(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "ok.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8")
    assert inspect_obb_labels(labels)["obb_like"] is True
    (labels / "bad.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    check = inspect_obb_labels(labels)
    assert check["obb_like"] is False
    assert check["bad_examples"]
