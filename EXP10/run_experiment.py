from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime import configure_matplotlib_env

configure_matplotlib_env("nnui2_exp10")


DATASET_NAME = "Bricks Detection Dataset"
DATASET_DOI = "10.5281/zenodo.18952529"
ZENODO_URL = "https://zenodo.org/records/18952529"
DEFAULT_SEARCH_ROOTS = [
    ROOT.parent / "cviceni",
    ROOT.parent / "cviceni" / "datasets",
    Path.home() / "Downloads",
]

VARIANTS: list[dict[str, object]] = [
    {"name": "V1_yolo26n_obb_img80", "requested_model": "yolo26n-obb.pt", "model_size": "n", "imgsz": 80, "epochs": 20, "batch": 8, "lr0": 0.01, "augment": "default"},
    {"name": "V2_yolo26n_obb_img240", "requested_model": "yolo26n-obb.pt", "model_size": "n", "imgsz": 240, "epochs": 20, "batch": 8, "lr0": 0.01, "augment": "default"},
    {"name": "V3_yolo26s_obb_img240", "requested_model": "yolo26s-obb.pt", "model_size": "s", "imgsz": 240, "epochs": 20, "batch": 8, "lr0": 0.01, "augment": "default"},
    {"name": "V4_yolo26s_obb_img320", "requested_model": "yolo26s-obb.pt", "model_size": "s", "imgsz": 320, "epochs": 20, "batch": 8, "lr0": 0.01, "augment": "default"},
    {"name": "V5_yolo26n_obb_img320_lr005_aug", "requested_model": "yolo26n-obb.pt", "model_size": "n", "imgsz": 320, "epochs": 20, "batch": 8, "lr0": 0.005, "augment": "mosaic=0.5, degrees=15"},
]

METRIC_FIELDS = [
    "variant",
    "requested_model",
    "actual_model",
    "imgsz",
    "epochs",
    "batch",
    "learning_rate",
    "augment",
    "status",
    "train_loss",
    "validation_loss",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "test_precision",
    "test_recall",
    "test_map50",
    "test_map50_95",
    "best_weights",
    "confusion_matrix",
    "test_confusion_matrix",
    "test_prediction",
    "command",
    "notes",
]


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {path}")
    return data


def resolve_dataset_path(yaml_path: Path, value: object) -> Path | None:
    if value is None:
        return None
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    data = load_yaml(yaml_path)
    base = Path(str(data.get("path", yaml_path.parent)))
    if not base.is_absolute():
        base = yaml_path.parent / base
    candidate = (base / raw).resolve()
    if candidate.exists():
        return candidate
    parts = list(raw.parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    fallback = (yaml_path.parent / Path(*parts)).resolve() if parts else candidate
    return fallback if fallback.exists() else candidate


def infer_label_dir(image_dir: Path | None) -> Path | None:
    if image_dir is None:
        return None
    parts = list(image_dir.parts)
    for token in ("images", "Images"):
        if token in parts:
            parts[parts.index(token)] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels"


def count_images(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return sum(1 for file in path.rglob("*") if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def count_labels(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return sum(1 for file in path.rglob("*.txt"))


def inspect_obb_labels(label_dir: Path | None) -> dict[str, object]:
    if label_dir is None or not label_dir.exists():
        return {"exists": False, "checked_files": 0, "non_empty_rows": 0, "obb_like": False, "bad_examples": []}
    checked = 0
    non_empty_rows = 0
    bad_examples: list[str] = []
    for file in sorted(label_dir.rglob("*.txt")):
        checked += 1
        for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            non_empty_rows += 1
            parts = stripped.split()
            if len(parts) != 9:
                bad_examples.append(f"{file}:{line_number}: expected 9 values, got {len(parts)}")
                break
            try:
                int(float(parts[0]))
                [float(value) for value in parts[1:]]
            except ValueError:
                bad_examples.append(f"{file}:{line_number}: non numeric OBB value")
                break
        if bad_examples:
            break
    return {
        "exists": True,
        "checked_files": checked,
        "non_empty_rows": non_empty_rows,
        "obb_like": checked > 0 and non_empty_rows > 0 and not bad_examples,
        "bad_examples": bad_examples,
    }


def looks_like_segmentation_yaml(yaml_path: Path, data: dict[str, Any]) -> bool:
    text = f"{yaml_path} {data.get('task', '')} {data.get('roboflow', '')}".lower()
    return "segment" in text or "segmentation" in text


def find_obb_yaml(search_roots: list[Path] = DEFAULT_SEARCH_ROOTS) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for yaml_path in list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml")):
            try:
                data = load_yaml(yaml_path)
            except Exception:
                continue
            if looks_like_segmentation_yaml(yaml_path, data):
                continue
            audit = inspect_dataset(str(yaml_path), allow_search=False)
            if audit.get("status") == "ok":
                score = 0
                lowered = str(yaml_path).lower()
                if "obb" in lowered:
                    score += 10
                if "brick" in lowered:
                    score += 5
                candidates.append((score, yaml_path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item[0], len(str(item[1]))))[0][1]


def inspect_dataset(data_path: str | None, allow_search: bool = True) -> dict[str, object]:
    if not data_path and allow_search:
        found = find_obb_yaml()
        if found:
            data_path = str(found)
    if not data_path:
        return {
            "status": "missing",
            "reason": "OBB dataset YAML path was not provided and no valid Bricks OBB data.yaml was found in configured search roots",
            "searched_roots": [str(path) for path in DEFAULT_SEARCH_ROOTS],
        }
    yaml_path = Path(data_path).expanduser().resolve()
    if not yaml_path.exists():
        return {"status": "missing", "reason": "dataset YAML path does not exist", "yaml_path": str(yaml_path)}
    try:
        data = load_yaml(yaml_path)
    except Exception as exc:
        return {"status": "invalid", "reason": f"cannot read dataset YAML: {exc!r}", "yaml_path": str(yaml_path)}
    if looks_like_segmentation_yaml(yaml_path, data):
        return {"status": "PARTIAL_NOT_OBB", "reason": "YAML/path metadata looks like a segmentation dataset, not the OBB branch", "yaml_path": str(yaml_path)}

    train = resolve_dataset_path(yaml_path, data.get("train"))
    val = resolve_dataset_path(yaml_path, data.get("val") or data.get("valid"))
    test = resolve_dataset_path(yaml_path, data.get("test"))
    train_labels = infer_label_dir(train)
    val_labels = infer_label_dir(val)
    test_labels = infer_label_dir(test)
    label_checks = {
        "train": inspect_obb_labels(train_labels),
        "val": inspect_obb_labels(val_labels),
        "test": inspect_obb_labels(test_labels),
    }
    names = data.get("names", [])
    nc = data.get("nc", len(names) if isinstance(names, (list, dict)) else None)
    has_splits = all(path is not None and path.exists() for path in [train, val, test])
    all_label_dirs = all(path is not None and path.exists() for path in [train_labels, val_labels, test_labels])
    obb_like = all(bool(check["obb_like"]) for check in label_checks.values())
    status = "ok" if has_splits and all_label_dirs and obb_like else "PARTIAL_NOT_OBB"
    reason = None
    if not has_splits:
        reason = "train/val/test image paths are missing"
    elif not all_label_dirs:
        reason = "train/val/test label paths are missing"
    elif not obb_like:
        reason = "at least one label split is not YOLO OBB 9-value format"
    return {
        "status": status,
        "dataset_name": DATASET_NAME,
        "doi": DATASET_DOI,
        "source": ZENODO_URL,
        "used_part": "OBB",
        "yaml_path": str(yaml_path),
        "nc": nc,
        "names": names,
        "train_images_path": str(train) if train else None,
        "val_images_path": str(val) if val else None,
        "test_images_path": str(test) if test else None,
        "train_labels_path": str(train_labels) if train_labels else None,
        "val_labels_path": str(val_labels) if val_labels else None,
        "test_labels_path": str(test_labels) if test_labels else None,
        "train_images": count_images(train),
        "val_images": count_images(val),
        "test_images": count_images(test),
        "train_labels": count_labels(train_labels),
        "val_labels": count_labels(val_labels),
        "test_labels": count_labels(test_labels),
        "label_checks": label_checks,
        "labels_have_9_values": obb_like,
        "reason": reason,
    }


def compatible_model_candidates(size: str) -> list[str]:
    if size == "s":
        return ["yolo26s-obb.pt", "yolo11s-obb.pt", "yolov8s-obb.pt"]
    return ["yolo26n-obb.pt", "yolo11n-obb.pt", "yolov8n-obb.pt"]


def resolve_yolo_model(variant: dict[str, object]) -> tuple[Any, str, str]:
    from ultralytics import YOLO  # type: ignore

    requested = str(variant["requested_model"])
    errors: list[str] = []
    for candidate in compatible_model_candidates(str(variant["model_size"])):
        try:
            return YOLO(candidate), candidate, "requested model used" if candidate == requested else f"fallback used because requested model {requested} was not loadable"
        except Exception as exc:
            errors.append(f"{candidate}: {exc!r}")
    raise RuntimeError("; ".join(errors))


def variant_command(variant: dict[str, object], data_path: str, project: Path, device: str, epochs: int | None = None) -> str:
    extra = ""
    if variant["name"] == "V5_yolo26n_obb_img320_lr005_aug":
        extra = " degrees=15 mosaic=0.5"
    epoch_text = f"--epochs {epochs}" if epochs else "--epochs 20"
    return (
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py "
        f"--data {data_path} --smoke {epoch_text} --device {device} "
        f"# saves into {project / str(variant['name'])}{extra}"
    )


def empty_metric_row(variant: dict[str, object], status: str, notes: str, data_path: str | None, project: Path, device: str, epochs: int | None = None) -> dict[str, object]:
    return {
        "variant": variant["name"],
        "requested_model": variant["requested_model"],
        "actual_model": "",
        "imgsz": variant["imgsz"],
        "epochs": epochs if epochs is not None else variant["epochs"],
        "batch": variant["batch"],
        "learning_rate": variant["lr0"],
        "augment": variant["augment"],
        "status": status,
        "train_loss": "",
        "validation_loss": "",
        "precision": "",
        "recall": "",
        "map50": "",
        "map50_95": "",
        "test_precision": "",
        "test_recall": "",
        "test_map50": "",
        "test_map50_95": "",
        "best_weights": "",
        "confusion_matrix": "",
        "test_confusion_matrix": "",
        "test_prediction": "",
        "command": variant_command(variant, data_path or "/path/to/bricks_obb/data.yaml", project, device, epochs),
        "notes": notes,
    }


def write_csv(rows: list[dict[str, object]], path: Path, fields: list[str] = METRIC_FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_yolo_yaml(dataset: dict[str, object], path: Path) -> Path:
    names = dataset.get("names", {})
    if isinstance(names, dict):
        name_lines = [f"  {key}: {value}" for key, value in names.items()]
    elif isinstance(names, list):
        name_lines = [f"  {index}: {value}" for index, value in enumerate(names)]
    else:
        name_lines = []
    lines = [
        f"path: {Path(str(dataset['train_images_path'])).parents[1]}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *name_lines,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_yolo_results_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def first_existing(row: dict[str, str], candidates: list[str]) -> str:
    normalized = {key.strip(): value.strip() for key, value in row.items()}
    for candidate in candidates:
        if candidate in normalized and normalized[candidate] != "":
            return normalized[candidate]
    return ""


def extract_metrics_from_run(save_dir: Path) -> dict[str, object]:
    result_row = read_yolo_results_csv(save_dir / "results.csv")
    weights = save_dir / "weights" / "best.pt"
    confusion_candidates = [
        save_dir / "confusion_matrix.png",
        save_dir / "confusion_matrix_normalized.png",
        save_dir / "val" / "confusion_matrix.png",
        save_dir / "test" / "confusion_matrix.png",
    ]
    prediction_candidates = sorted(save_dir.rglob("*.jpg")) + sorted(save_dir.rglob("*.png"))
    prediction = next((path for path in prediction_candidates if "pred" in path.name.lower()), None)
    return {
        "train_loss": first_existing(result_row, ["train/box_loss", "train/obb_loss", "train/dfl_loss"]),
        "validation_loss": first_existing(result_row, ["val/box_loss", "val/obb_loss", "val/dfl_loss"]),
        "precision": first_existing(result_row, ["metrics/precision(B)", "metrics/precision(OBB)", "metrics/precision"]),
        "recall": first_existing(result_row, ["metrics/recall(B)", "metrics/recall(OBB)", "metrics/recall"]),
        "map50": first_existing(result_row, ["metrics/mAP50(B)", "metrics/mAP50(OBB)", "metrics/mAP50"]),
        "map50_95": first_existing(result_row, ["metrics/mAP50-95(B)", "metrics/mAP50-95(OBB)", "metrics/mAP50-95"]),
        "best_weights": str(weights) if weights.exists() else "",
        "confusion_matrix": str(next((path for path in confusion_candidates if path.exists()), "")),
        "test_prediction": str(prediction or ""),
    }


def metric_attr(root: Any, path: str) -> float | None:
    current = root
    for part in path.split("."):
        if current is None:
            return None
        current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
    if current is None:
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.5g}"


def extract_test_metrics(test_result: Any, test_save_dir: Path) -> dict[str, object]:
    metric_root = getattr(test_result, "box", None) or getattr(test_result, "obb", None)
    prediction_candidates = sorted(test_save_dir.rglob("*.jpg")) + sorted(test_save_dir.rglob("*.png"))
    prediction = next((path for path in prediction_candidates if "pred" in path.name.lower()), None)
    confusion_candidates = [
        test_save_dir / "confusion_matrix.png",
        test_save_dir / "confusion_matrix_normalized.png",
    ]
    return {
        "test_precision": format_metric(metric_attr(metric_root, "mp")),
        "test_recall": format_metric(metric_attr(metric_root, "mr")),
        "test_map50": format_metric(metric_attr(metric_root, "map50")),
        "test_map50_95": format_metric(metric_attr(metric_root, "map")),
        "test_confusion_matrix": str(next((path for path in confusion_candidates if path.exists()), "")),
        "test_prediction": str(prediction or ""),
    }


def run_yolo_variant(variant: dict[str, object], data_path: str, project: Path, device: str, epochs: int) -> dict[str, object]:
    model, actual_model, model_note = resolve_yolo_model(variant)
    run_name = str(variant["name"])
    train_kwargs = {
        "data": data_path,
        "imgsz": int(variant["imgsz"]),
        "epochs": epochs,
        "batch": int(variant["batch"]),
        "lr0": float(variant["lr0"]),
        "project": str(project),
        "name": run_name,
        "task": "obb",
        "device": device,
        "exist_ok": True,
    }
    if variant["name"] == "V5_yolo26n_obb_img320_lr005_aug":
        train_kwargs.update({"degrees": 15, "mosaic": 0.5})
    train_result = model.train(**train_kwargs)
    save_dir = Path(getattr(train_result, "save_dir", project / run_name))
    test_metrics: dict[str, object] = {}
    try:
        test_result = model.val(data=data_path, split="test", task="obb", device=device, project=str(project), name=f"{run_name}_test", exist_ok=True)
        test_save_dir = Path(getattr(test_result, "save_dir", project / f"{run_name}_test"))
        test_metrics = extract_test_metrics(test_result, test_save_dir)
    except Exception as exc:
        test_metrics = {"notes": f"{model_note}; test split evaluation failed: {exc!r}"}
    row = empty_metric_row(variant, "completed", model_note, data_path, project, device, epochs)
    row["actual_model"] = actual_model
    row.update(extract_metrics_from_run(save_dir))
    row.update(test_metrics)
    return row


def select_best(rows: list[dict[str, object]]) -> dict[str, object] | None:
    completed = [row for row in rows if row.get("status") == "completed" and str(row.get("test_map50_95") or row.get("map50_95", "")) not in {"", "nan"}]
    if not completed:
        return None
    return max(
        completed,
        key=lambda row: (
            float(row.get("test_map50_95") or row.get("map50_95") or 0),
            float(row.get("test_map50") or row.get("map50") or 0),
            float(row.get("test_precision") or row.get("precision") or 0),
        ),
    )


def copy_best_artifacts(best: dict[str, object] | None, results: Path) -> None:
    if not best:
        return
    weights = Path(str(best.get("best_weights", "")))
    confusion = Path(str(best.get("test_confusion_matrix") or best.get("confusion_matrix", "")))
    prediction = Path(str(best.get("test_prediction", "")))
    if weights.exists():
        shutil.copy2(weights, results / "best_model.pt")
    if confusion.exists():
        shutil.copy2(confusion, results / "best_confusion_matrix.png")
    if prediction.exists():
        shutil.copy2(prediction, results / "test_prediction_best.png")


def write_report(root_report: Path, exp_report: Path, rows: list[dict[str, object]], dataset: dict[str, object], best: dict[str, object] | None, data_path: str | None, device: str, run_mode: str) -> None:
    dataset_status = str(dataset.get("status"))
    yaml_path = dataset.get("yaml_path", data_path or "nenalezeno")
    reason = dataset.get("reason") or ""
    best_lines = (
        [
            f"- Nejlepsi varianta: `{best['variant']}`.",
            f"- Kriterium: nejvyssi test `mAP50-95 = {best.get('test_map50_95') or best.get('map50_95')}`.",
            f"- Best weights: `{best['best_weights']}`.",
        ]
        if best
        else ["- Nejlepsi model nelze vybrat, protoze nejsou dostupne realne metriky `mAP50-95`."]
    )
    variant_table = [
        "| Varianta | Pozadovany model | Realny model | Image size | Epochs | Batch | LR | Augmentace | Hardware |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    result_table = [
        "| Varianta | Val P | Val R | Val mAP50 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test mAP50-95 | Train loss | Validation loss | Stav |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        variant_table.append(
            f"| `{row['variant']}` | `{row['requested_model']}` | `{row['actual_model'] or '-'}` | {row['imgsz']} | {row['epochs']} | {row['batch']} | {row['learning_rate']} | {row['augment']} | {device} |"
        )
        result_table.append(
            f"| `{row['variant']}` | {row['precision'] or '-'} | {row['recall'] or '-'} | {row['map50'] or '-'} | {row['map50_95'] or '-'} | {row['test_precision'] or '-'} | {row['test_recall'] or '-'} | {row['test_map50'] or '-'} | {row['test_map50_95'] or '-'} | {row['train_loss'] or '-'} | {row['validation_loss'] or '-'} | {row['status']} |"
        )
    lines = [
        "# NNUI2 - Cviceni 10: YOLO OBB detekce",
        "",
        "## 1. Cil experimentu",
        "Cilem je porovnat pet variant YOLO OBB detekce nad OBB casti Bricks Detection Datasetu a vybrat nejlepsi model podle `mAP50-95`.",
        "",
        "## 2. Dataset",
        f"- Dataset: `{DATASET_NAME}` ze Zenodo.",
        f"- DOI / zdroj: `{DATASET_DOI}`, `{ZENODO_URL}`.",
        "- Pouzita cast: `OBB`.",
        "- Segmentation cast neni pouzita, protoze EXP10 resi detekci orientovanych bounding boxu, ne segmentacni masky.",
        f"- YAML: `{yaml_path}`.",
        f"- Stav auditu datasetu: `{dataset_status}`. {reason}",
        f"- Pocet trid: `{dataset.get('nc', 'nezname')}`; nazvy: `{dataset.get('names', [])}`.",
        f"- Pocet obrazku train/val/test: `{dataset.get('train_images', 'nezname')}/{dataset.get('val_images', 'nezname')}/{dataset.get('test_images', 'nezname')}`.",
        f"- Pocet labelu train/val/test: `{dataset.get('train_labels', 'nezname')}/{dataset.get('val_labels', 'nezname')}/{dataset.get('test_labels', 'nezname')}`.",
        f"- Potvrzeni OBB formatu 9 hodnot: `{dataset.get('labels_have_9_values', False)}`.",
        "",
        "## 3. Varianty modelu",
        *variant_table,
        "",
        "## 4. Trenovani",
        f"- Rezim behu: `{run_mode}`.",
        "- Skript podporuje `--data PATH`, `--audit-only`, `--smoke` a `--epochs N`.",
        "- Pokud presny YOLO26 model neni dostupny v ultralytics, skript zkusi kompatibilni OBB model stejne velikosti (`yolo11*-obb.pt`, potom `yolov8*-obb.pt`) a realny model zapise do CSV/reportu.",
        "- Vystupy YOLO se ukladaji do `EXP10/results/yolo_runs/`.",
        "",
        "## 5. Vysledky",
        "Tabulka uvadi validacni metriky z trenovani a samostatnou evaluaci nad test splitem. Nejlepsi model se vybira podle test `mAP50-95`, pokud je k dispozici.",
        *result_table,
        "",
        "## 6. Nejlepsi model",
        *best_lines,
        "",
        "## 7. Confusion matrix a ukazka detekce",
        "- Pokud YOLO test evaluace vytvori confusion matrix, nejlepsi se kopiruje do `EXP10/results/best_confusion_matrix.png`.",
        "- Pokud YOLO test evaluace vytvori predikcni obrazek, nejlepsi se kopiruje do `EXP10/results/test_prediction_best.png`.",
        "",
        "## 8. Diskuze",
        "Mensi image size `80` slouzi hlavne jako rychly baseline/smoke. Varianty `240` a `320` by mely lepe zachytit polohu rohu OBB, ale jsou pomalejsi. Vetsi model `s` muze mit lepsi kapacitu nez `n`, ale muze byt narocnejsi na CPU/GPU. Varianta V5 meni learning rate a augmentaci, aby bylo videt, zda pomuze robustnosti.",
        "",
        "## 9. Zaver",
        "Report neobsahuje vymyslene metriky. Pokud je dataset nebo trenink nedostupny, radky zustavaji `not_run` nebo `failed` s prazdnymi metrikami. Pokud probehl jen smoke beh, stav experimentu je `PASS_WITH_LIMITATIONS`.",
        "",
    ]
    root_report.write_text("\n".join(lines), encoding="utf-8")
    exp_report.parent.mkdir(parents=True, exist_ok=True)
    exp_report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to Bricks Detection Dataset OBB data.yaml")
    parser.add_argument("--audit-only", action="store_true", help="Inspect dataset and write not_run rows without training")
    parser.add_argument("--smoke", action="store_true", help="Run all selected variants as a short smoke run")
    parser.add_argument("--run", action="store_true", help="Backward compatible alias for --smoke")
    parser.add_argument("--only", default=None, help="Run only one variant by name")
    parser.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu, 0")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    results = base / "results"
    yolo_project = results / "yolo_runs"
    results.mkdir(parents=True, exist_ok=True)

    dataset = inspect_dataset(args.data)
    data_path = str(dataset.get("yaml_path") or args.data or "")
    yolo_data_path = data_path
    if dataset.get("status") == "ok":
        yolo_data_path = str(write_runtime_yolo_yaml(dataset, results / "bricks_obb_runtime.yaml"))
    selected_variants = [variant for variant in VARIANTS if args.only in {None, variant["name"]}]
    if not selected_variants:
        raise SystemExit(f"Unknown variant for --only: {args.only}")

    should_train = (args.smoke or args.run) and not args.audit_only
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke else 20)
    rows: list[dict[str, object]] = []
    if dataset.get("status") != "ok":
        status = "not_run" if dataset.get("status") in {"missing", "invalid"} else str(dataset.get("status"))
        note = f"dataset audit failed: {dataset.get('reason') or dataset.get('status')}"
        rows = [empty_metric_row(variant, status, note, yolo_data_path, yolo_project, args.device, epochs) for variant in selected_variants]
    elif not should_train:
        rows = [empty_metric_row(variant, "not_run", "audit-only; training was not requested", yolo_data_path, yolo_project, args.device, epochs) for variant in selected_variants]
    else:
        for variant in selected_variants:
            try:
                rows.append(run_yolo_variant(variant, yolo_data_path, yolo_project, args.device, epochs))
            except Exception as exc:
                rows.append(empty_metric_row(variant, "failed", f"YOLO run failed: {exc!r}", yolo_data_path, yolo_project, args.device, epochs))

    best = select_best(rows)
    copy_best_artifacts(best, results)
    write_csv(rows, results / "metrics.csv")
    write_csv(rows, results / "variant_summary.csv")
    (results / "dataset_check.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (results / "results.json").write_text(
        json.dumps({"status": "PASS_WITH_LIMITATIONS" if should_train and best else "PARTIAL", "dataset": dataset, "variants": VARIANTS, "rows": rows, "best": best}, indent=2),
        encoding="utf-8",
    )
    run_mode = "smoke" if should_train and epochs == 1 else ("train" if should_train else "audit-only")
    write_report(ROOT / "experiment_10.md", base / "report" / "report.md", rows, dataset, best, data_path or args.data, args.device, run_mode)

    print("EXP10 completed")
    print(f"Dataset status: {dataset.get('status')}")
    print(f"Dataset YAML: {dataset.get('yaml_path')}")
    print(f"Metrics: {results / 'metrics.csv'}")
    print(f"Report: {ROOT / 'experiment_10.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
