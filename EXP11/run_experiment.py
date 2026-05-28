from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime import configure_matplotlib_env

configure_matplotlib_env("nnui2_exp11")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 1111
IMAGE_SIZE = 128
MAX_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAIN_LIMIT = 64
VAL_LIMIT = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POTATOES_DATASET_DIR = ROOT.parent / "cviceni" / "Potatoes_seg"
SEGMENTACE_TXT = ROOT.parent / "cviceni" / "segmentace.txt"
CLASS_NAMES = ["potatoes-"]
RUN_STATUS = "PASS_WITH_LIMITATIONS"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


class TinySegNet(nn.Module):
    """Small encoder-decoder CNN used for a fast local segmentation smoke run."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PotatoesSegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[Sample], image_size: int = IMAGE_SIZE) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image, mask = load_image_and_mask(sample.image_path, sample.label_path, self.image_size)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask[None, :, :]).float()
        return image_tensor, mask_tensor


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_yaml_simple(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return data


def find_samples(split: str, dataset_dir: Path = POTATOES_DATASET_DIR) -> list[Sample]:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    samples: list[Sample] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append(Sample(image_path=image_path, label_path=label_path))
    return samples


def parse_yolo_polygons(label_path: Path, width: int, height: int) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) < 7:
            continue
        coords = [float(value) for value in values[1:]]
        if len(coords) % 2 != 0:
            continue
        polygon = [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)]
        polygons.append(polygon)
    return polygons


def load_image_and_mask(image_path: Path, label_path: Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as raw:
        rgb = raw.convert("RGB")
        width, height = rgb.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in parse_yolo_polygons(label_path, width, height):
            if len(polygon) >= 3:
                draw.polygon(polygon, outline=1, fill=1)
        rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    image_array = np.asarray(rgb, dtype=np.float32) / 255.0
    mask_array = np.asarray(mask, dtype=np.float32)
    mask_array = (mask_array > 0).astype(np.float32)
    return image_array, mask_array


def inspect_dataset(dataset_dir: Path = POTATOES_DATASET_DIR) -> dict[str, Any]:
    yaml_path = dataset_dir / "data.yaml"
    info: dict[str, Any] = {
        "path": str(dataset_dir),
        "yaml_path": str(yaml_path),
        "exists": dataset_dir.exists(),
        "has_data_yaml": yaml_path.exists(),
        "has_images": False,
        "has_masks_or_annotations": False,
        "annotation_format": "YOLO segmentation polygon labels",
        "splits": {},
        "nc": None,
        "names": [],
        "status": "PARTIAL",
        "reason": "",
    }
    if yaml_path.exists():
        try:
            yaml_data = read_yaml_simple(yaml_path)
            info["nc"] = yaml_data.get("nc")
            info["names"] = yaml_data.get("names", [])
            info["yaml"] = yaml_data
        except Exception as exc:
            info["reason"] = f"data.yaml could not be read: {exc!r}"
            return info

    total_images = 0
    total_labels = 0
    polygon_files_total = 0
    for split in ["train", "valid", "test"]:
        samples = find_samples(split, dataset_dir) if (dataset_dir / split / "images").exists() else []
        image_count = len(list((dataset_dir / split / "images").glob("*"))) if (dataset_dir / split / "images").exists() else 0
        label_count = len(list((dataset_dir / split / "labels").glob("*.txt"))) if (dataset_dir / split / "labels").exists() else 0
        total_images += image_count
        total_labels += label_count
        checked = 0
        polygon_files = 0
        for sample in samples[:20]:
            checked += 1
            polygons = parse_yolo_polygons(sample.label_path, width=100, height=100)
            if polygons:
                polygon_files += 1
        polygon_files_total += polygon_files
        info["splits"][split] = {
            "images_path": str(dataset_dir / split / "images"),
            "labels_path": str(dataset_dir / split / "labels"),
            "images": image_count,
            "labels": label_count,
            "paired_samples": len(samples),
            "checked_labels": checked,
            "checked_labels_with_polygons": polygon_files,
        }

    info["has_images"] = total_images > 0
    info["has_masks_or_annotations"] = total_labels > 0 and polygon_files_total > 0
    info["total_images"] = total_images
    info["total_labels"] = total_labels
    if info["exists"] and info["has_data_yaml"] and info["has_images"] and info["has_masks_or_annotations"]:
        info["status"] = "OK"
        info["reason"] = "Potatoes_seg contains train/valid/test images and YOLO polygon segmentation annotations."
    else:
        info["reason"] = "Dataset is missing images, labels, data.yaml, or valid segmentation polygons."
    return info


def make_loader(samples: list[Sample], batch_size: int, shuffle: bool, seed: int) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(PotatoesSegDataset(samples), batch_size=batch_size, shuffle=shuffle, generator=generator)


def segmentation_metrics(logits: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= 0.5
    truth = masks >= 0.5
    intersection = torch.logical_and(preds, truth).sum().item()
    union = torch.logical_or(preds, truth).sum().item()
    pred_sum = preds.sum().item()
    truth_sum = truth.sum().item()
    correct = (preds == truth).sum().item()
    total = truth.numel()
    iou = intersection / union if union else 1.0
    dice = (2 * intersection) / (pred_sum + truth_sum) if (pred_sum + truth_sum) else 1.0
    return {"pixel_accuracy": correct / total, "iou": iou, "dice": dice}


def evaluate(model: nn.Module, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], criterion: nn.Module) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    totals = {"pixel_accuracy": 0.0, "iou": 0.0, "dice": 0.0}
    batches = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)
            losses.append(float(loss.item()))
            metrics = segmentation_metrics(logits.cpu(), masks.cpu())
            for key in totals:
                totals[key] += metrics[key]
            batches += 1
    if batches == 0:
        return {"loss": float("nan"), "pixel_accuracy": float("nan"), "iou": float("nan"), "dice": float("nan")}
    return {
        "loss": float(np.mean(losses)),
        "pixel_accuracy": totals["pixel_accuracy"] / batches,
        "iou": totals["iou"] / batches,
        "dice": totals["dice"] / batches,
    }


def train_model(train_samples: list[Sample], val_samples: list[Sample], test_samples: list[Sample]) -> tuple[nn.Module, list[dict[str, float]], dict[str, float]]:
    set_seed()
    model = TinySegNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(train_samples[:TRAIN_LIMIT], BATCH_SIZE, True, SEED)
    val_loader = make_loader(val_samples[:VAL_LIMIT], BATCH_SIZE, False, SEED)
    history: list[dict[str, float]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val = evaluate(model, val_loader, criterion)
        row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        if val["loss"] < best_val:
            best_val = val["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, make_loader(test_samples, BATCH_SIZE, False, SEED), criterion)
    return model.cpu(), history, test


def save_metrics_csv(history: list[dict[str, float]], test: dict[str, float], path: Path) -> None:
    fields = ["epoch", "train_loss", "val_loss", "val_pixel_accuracy", "val_iou", "val_dice", "test_loss", "test_pixel_accuracy", "test_iou", "test_dice"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_pixel_accuracy": row["val_pixel_accuracy"],
                    "val_iou": row["val_iou"],
                    "val_dice": row["val_dice"],
                    "test_loss": test["loss"],
                    "test_pixel_accuracy": test["pixel_accuracy"],
                    "test_iou": test["iou"],
                    "test_dice": test["dice"],
                }
            )


def save_segmentation_samples(model: nn.Module, samples: list[Sample], path: Path, count: int = 3) -> None:
    model.eval()
    selected = samples[:count]
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    axes = np.atleast_2d(axes)
    with torch.no_grad():
        for row, sample in enumerate(selected):
            image, mask = load_image_and_mask(sample.image_path, sample.label_path, IMAGE_SIZE)
            tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))[None]).float()
            pred = torch.sigmoid(model(tensor))[0, 0].numpy()
            pred_bin = pred >= 0.5
            axes[row, 0].imshow(image)
            axes[row, 0].set_title("vstup")
            axes[row, 1].imshow(mask, cmap="gray")
            axes[row, 1].set_title("maska")
            axes[row, 2].imshow(pred_bin, cmap="gray")
            axes[row, 2].set_title("predikce")
            for col in range(3):
                axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(
    root_report: Path,
    exp_report: Path,
    dataset: dict[str, Any],
    history: list[dict[str, float]],
    test: dict[str, float],
    ran_training: bool,
) -> None:
    segmentace_note = "Soubor `segmentace.txt` obsahuje kostru pro YOLO26/Ultralytics segmentaci s `YOLO('yolo26n-seg.pt')`, `model.train(data='...data.yaml')`, validaci a predikci. EXP11 z nej prebiral hlavni informaci, ze lokalni data jsou YOLO segmentacni dataset s `data.yaml`; pro bezpecny kratky smoke beh je zde pouzita mala PyTorch CNN nad stejnymi polygonovymi anotacemi."
    status = RUN_STATUS if ran_training else "PARTIAL"
    final_epoch = history[-1] if history else {}
    lines = [
        "# NNUI2 - Cviceni 11: segmentace Potatoes_seg",
        "",
        "## 1. Stav experimentu",
        f"- Stav: `{status}`.",
        "- Duvod omezeni: beh je kratky smoke test male segmentacni CNN, nikoli plny dlouhy YOLO26-seg trenink.",
        "",
        "## 2. Dataset",
        f"- Hlavni dataset: `{dataset['path']}`.",
        f"- YAML: `{dataset['yaml_path']}`.",
        f"- Struktura: `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`.",
        f"- Obsahuje obrazky: `{dataset['has_images']}`.",
        f"- Obsahuje masky/anotace: `{dataset['has_masks_or_annotations']}`.",
        f"- Format anotaci: `{dataset['annotation_format']}`.",
        f"- Pocet trid: `{dataset.get('nc')}`; nazvy trid: `{dataset.get('names')}`.",
        f"- Pocet obrazku train/valid/test: `{dataset['splits']['train']['images']}/{dataset['splits']['valid']['images']}/{dataset['splits']['test']['images']}`.",
        "- Flower classification dataset `cviceni/dataset/` neni v EXP11 pouzit.",
        "",
        "## 3. Metodika",
        "YOLO polygon labely se prevadeji na binarni masky vykreslenim polygonu do masky stejne velikosti jako vstupni obrazek. Obrazky a masky se nasledne resizeuji na `128x128`.",
        f"Trenovani pouziva pripraveny split datasetu: maximalne `{TRAIN_LIMIT}` trenovacich obrazku, `{VAL_LIMIT}` validacnich obrazku a vsechny test obrazky. Pixely jsou normalizovane do rozsahu `0-1`.",
        segmentace_note,
        "",
        "## 4. Model",
        "Pouzita je mala PyTorch encoder-decoder CNN `TinySegNet`: dve konvolucni casti s poolingem a decoder s bilinear upsamplingem. Vystupem je jedna binarni maska pro tridu potatoes.",
        f"- Loss: `BCEWithLogitsLoss`.",
        f"- Optimizer: `Adam`, learning rate `{LEARNING_RATE}`.",
        f"- Epochs: `{MAX_EPOCHS}`.",
        f"- Batch size: `{BATCH_SIZE}`.",
        f"- Device: `{DEVICE}`.",
        "",
        "## 5. Vysledky",
        f"- Posledni train loss: `{final_epoch.get('train_loss', 'n/a')}`.",
        f"- Posledni validation loss: `{final_epoch.get('val_loss', 'n/a')}`.",
        f"- Test loss: `{test.get('loss', 'n/a')}`.",
        f"- Test pixel accuracy: `{test.get('pixel_accuracy', 'n/a')}`.",
        f"- Test IoU: `{test.get('iou', 'n/a')}`.",
        f"- Test Dice: `{test.get('dice', 'n/a')}`.",
        "- Detailni metriky jsou v `EXP11/results/metrics.csv`.",
        "",
        "## 6. Ukazky segmentace",
        "Ukazky obsahuji vstup, referencni masku z YOLO polygonu a predikci male CNN.",
        "",
        "![Ukazky segmentace](EXP11/results/segmentation_samples.png)",
        "",
        "## 7. Diskuze",
        "Dataset je vhodny pro segmentaci: obsahuje realne obrazky, pripravene train/valid/test rozdeleni, `data.yaml` a YOLO polygonove anotace. Neobsahuje samostatne bitmapove masky, ale ty lze korektne odvodit z polygonu bez vytvareni falesnych masek.",
        "Kvalita segmentace je limitovana kratkym smoke treninkem a malou architekturou. Metriky proto overuji funkcnost pipeline, ne finalni produkcni presnost.",
        "Pro plnohodnotny experiment podle `segmentace.txt` by dalsi krok byl spustit YOLO26-seg/Ultralytics trenink nad stejnym `data.yaml`, ulozit mask mAP metriky a porovnat vice variant.",
        "",
        "## 8. Zaver",
        "EXP11 je sjednocen s lokalnim `Potatoes_seg/` datasetem a jasne resi segmentaci, ne klasifikaci. Vznikly realne metriky kratkeho PyTorch CNN smoke behu, ulozeny model a ukazky segmentace. Stav je `PASS_WITH_LIMITATIONS`, protoze neprobehl dlouhy YOLO26-seg trenink.",
        "",
    ]
    root_report.write_text("\n".join(lines), encoding="utf-8")
    exp_report.parent.mkdir(parents=True, exist_ok=True)
    exp_report.write_text("\n".join(lines).replace("EXP11/results/", "../results/"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="Only inspect Potatoes_seg without training the smoke model")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)

    dataset = inspect_dataset()
    train_samples = find_samples("train")
    val_samples = find_samples("valid")
    test_samples = find_samples("test")
    ran_training = False
    history: list[dict[str, float]] = []
    test = {"loss": float("nan"), "pixel_accuracy": float("nan"), "iou": float("nan"), "dice": float("nan")}

    if dataset["status"] == "OK" and not args.audit_only:
        model, history, test = train_model(train_samples, val_samples, test_samples)
        torch.save(
            {
                "model_state": model.state_dict(),
                "model": "TinySegNet",
                "status": RUN_STATUS,
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "test_metrics": test,
            },
            results / "best_model.pt",
        )
        save_segmentation_samples(model, test_samples, results / "segmentation_samples.png")
        ran_training = True

    save_metrics_csv(history, test, results / "metrics.csv")
    (results / "results.json").write_text(
        json.dumps(
            {
                "status": RUN_STATUS if ran_training else "PARTIAL",
                "dataset": dataset,
                "segmentace_txt": {
                    "path": str(SEGMENTACE_TXT),
                    "exists": SEGMENTACE_TXT.exists(),
                    "used_as": "YOLO26-seg instruction/source material; local smoke run uses the same data.yaml and segmentation labels.",
                },
                "model": "TinySegNet",
                "training": {
                    "epochs": MAX_EPOCHS if ran_training else 0,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "train_limit": TRAIN_LIMIT,
                    "val_limit": VAL_LIMIT,
                    "device": str(DEVICE),
                },
                "history": history,
                "test_metrics": test,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(ROOT / "experiment_11.md", base / "report" / "report.md", dataset, history, test, ran_training)

    print("EXP11 completed" if ran_training else "EXP11 audit completed")
    print(f"Dataset: {POTATOES_DATASET_DIR}")
    print(f"Dataset status: {dataset['status']}")
    print(f"Metrics: {results / 'metrics.csv'}")
    print(f"Report: {ROOT / 'experiment_11.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
