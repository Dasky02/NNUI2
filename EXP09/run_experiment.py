from __future__ import annotations

import csv
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime import configure_matplotlib_env

configure_matplotlib_env("nnui2_exp09")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 9090
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 8
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOWER_DATASET_DIR = ROOT.parent / "cviceni" / "dataset"
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
IMAGE_SIZE = 64
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RUN_STATUS = "PASS_WITH_LIMITATIONS"


ARCHITECTURES: list[dict[str, object]] = [
    {
        "name": "A1_conv1_pool_fc",
        "channels": [8],
        "kernel_sizes": [3],
        "pool_after": [0],
        "dropout": 0.0,
        "activation": "relu",
        "fc_hidden": 48,
        "description": "1x Conv + Pool + FC",
    },
    {
        "name": "A2_conv2_pool_fc",
        "channels": [8, 16],
        "kernel_sizes": [3, 3],
        "pool_after": [1],
        "dropout": 0.0,
        "activation": "relu",
        "fc_hidden": 48,
        "description": "2x Conv + Pool + FC",
    },
    {
        "name": "A3_conv3_pool_fc",
        "channels": [8, 16, 16],
        "kernel_sizes": [3, 3, 3],
        "pool_after": [1],
        "dropout": 0.0,
        "activation": "relu",
        "fc_hidden": 48,
        "description": "3x Conv + Pool + FC",
    },
    {
        "name": "A4_conv2_dropout_fc",
        "channels": [8, 16],
        "kernel_sizes": [3, 3],
        "pool_after": [1],
        "dropout": 0.25,
        "activation": "relu",
        "fc_hidden": 48,
        "description": "2x Conv + Dropout + FC",
    },
    {
        "name": "A5_conv3_dropout_bigfc_tanh",
        "channels": [8, 16, 24],
        "kernel_sizes": [3, 3, 3],
        "pool_after": [1],
        "dropout": 0.30,
        "activation": "tanh",
        "fc_hidden": 96,
        "description": "3x Conv + Dropout + vetsi FC",
    },
]


class FlexibleCNN(nn.Module):
    def __init__(
        self,
        config: dict[str, object],
        num_classes: int = len(CLASS_NAMES),
        input_shape: tuple[int, int, int] = (3, IMAGE_SIZE, IMAGE_SIZE),
    ) -> None:
        super().__init__()
        activation_name = str(config["activation"])
        if activation_name == "relu":
            activation_cls: type[nn.Module] = nn.ReLU
        elif activation_name == "tanh":
            activation_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        channels = [int(value) for value in config["channels"]]
        kernels = [int(value) for value in config["kernel_sizes"]]
        pool_after = {int(value) for value in config["pool_after"]}
        if len(channels) != len(kernels):
            raise ValueError("Each convolutional layer must have one kernel size")

        layers: list[nn.Module] = []
        in_channels = input_shape[0]
        for index, (out_channels, kernel) in enumerate(zip(channels, kernels, strict=True)):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2))
            layers.append(activation_cls())
            if index in pool_after:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feature_dim = int(np.prod(self.features(dummy).shape[1:]))

        dropout = float(config["dropout"])
        fc_hidden = int(config["fc_hidden"])
        classifier: list[nn.Module] = [nn.Flatten(), nn.Linear(feature_dim, fc_hidden), activation_cls()]
        if dropout > 0:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(fc_hidden, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_flower_images(dataset_dir: Path = FLOWER_DATASET_DIR) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Flower dataset directory was not found: {dataset_dir}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    counts: dict[str, int] = {}
    skipped = 0
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required class directory was not found: {class_dir}")
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        counts[class_name] = len(files)
        for path in files:
            try:
                with Image.open(path) as image:
                    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                    array = np.asarray(resized, dtype=np.float32) / 255.0
            except Exception:
                skipped += 1
                continue
            images.append(np.transpose(array, (2, 0, 1)))
            labels.append(label)

    if not images:
        raise RuntimeError(f"No readable flower images were loaded from {dataset_dir}")
    if min(counts.values()) < 2:
        raise RuntimeError(f"Each class needs at least two images for stratified split: {counts}")

    counts["_skipped_unreadable"] = skipped
    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int64), CLASS_NAMES.copy(), counts


def prepare_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, object],
]:
    images, labels, class_names, class_counts = load_flower_images()
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        np.arange(len(labels)),
        labels,
        test_size=0.2,
        random_state=43,
        stratify=labels,
    )
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=0.25,
        random_state=43,
        stratify=y_train_val,
    )
    train_mean = images[train_idx].mean(axis=(0, 2, 3), keepdims=True)
    train_std = images[train_idx].std(axis=(0, 2, 3), keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    images = (images - train_mean) / train_std
    dataset_info = {
        "name": "Local flower classification dataset",
        "path": str(FLOWER_DATASET_DIR),
        "classes": len(class_names),
        "class_names": class_names,
        "class_counts": class_counts,
        "images": int(len(images)),
        "image_size": f"3x{IMAGE_SIZE}x{IMAGE_SIZE} RGB",
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "split": "60/20/20 stratified",
        "preprocessing": "RGB conversion, resize to 64x64, pixel scaling to 0-1, channel-wise normalization using train split mean/std",
        "normalization_mean": np.ravel(train_mean).round(6).tolist(),
        "normalization_std": np.ravel(train_std).round(6).tolist(),
    }
    return images, train_idx, val_idx, test_idx, y_train, y_val, y_test, class_names, dataset_info


def predict(model: nn.Module, X: np.ndarray, device: torch.device = DEVICE) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.argmax(logits, dim=1).cpu().numpy()


def evaluate_loss(model: nn.Module, X: np.ndarray, y: np.ndarray, criterion: nn.Module, device: torch.device = DEVICE) -> float:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        yb = torch.tensor(y, dtype=torch.long, device=device)
        return float(criterion(model(xb), yb).item())


def train_and_score(
    images: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, object],
    seed: int,
    device: torch.device = DEVICE,
) -> tuple[dict[str, object], FlexibleCNN, np.ndarray, dict[str, list[float]]]:
    set_seed(seed)
    model = FlexibleCNN(config, num_classes=len(CLASS_NAMES), input_shape=tuple(images.shape[1:])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(images[train_idx], dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = -1.0
    stale = 0
    curves = {"train_loss": [], "val_loss": []}

    for _epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(yb)
            seen += len(yb)
        curves["train_loss"].append(epoch_loss / max(seen, 1))
        curves["val_loss"].append(evaluate_loss(model, images[val_idx], y_val, criterion, device))
        val_acc = float(accuracy_score(y_val, predict(model, images[val_idx], device)))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.to(device)
    pred = predict(model, images[test_idx], device)
    test_accuracy = float(accuracy_score(y_test, pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    row: dict[str, object] = {
        "architecture": config["name"],
        "description": config["description"],
        "conv_layers": len(config["channels"]),
        "filters": "-".join(str(value) for value in config["channels"]),
        "kernel_sizes": "-".join(str(value) for value in config["kernel_sizes"]),
        "pool_after": str(config["pool_after"]),
        "dropout": float(config["dropout"]),
        "activation": config["activation"],
        "fc_hidden": int(config["fc_hidden"]),
        "seed": seed,
        "val_accuracy": float(accuracy_score(y_val, predict(model, images[val_idx], device))),
        "accuracy": test_accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "test_loss": evaluate_loss(model, images[test_idx], y_test, criterion, device),
        "epochs": len(curves["train_loss"]),
    }
    return row, model.cpu(), pred, curves


def save_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predikce")
    plt.ylabel("Skutecnost")
    plt.title("Confusion matrix nejlepsi CNN architektury")
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(int(cm[row, col])), ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_architecture_comparison(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["architecture"]).split("_")[0] for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    f1_scores = [float(row["f1_macro"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, accuracy, width=width, label="accuracy")
    plt.bar(x + width / 2, f1_scores, width=width, label="F1 macro")
    plt.xticks(x, labels)
    lower = max(0.0, min(accuracy + f1_scores) - 0.08)
    upper = min(1.0, max(accuracy + f1_scores) + 0.04)
    plt.ylim(lower, upper)
    plt.ylabel("Score")
    plt.xlabel("Architektura")
    plt.title("Porovnani CNN architektur")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_loss_curves(curves_by_arch: dict[str, dict[str, list[float]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for name, curves in curves_by_arch.items():
        label = name.split("_")[0]
        axes[0].plot(curves["train_loss"], label=label)
        axes[1].plot(curves["val_loss"], label=label)
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epocha")
        ax.set_ylabel("Cross entropy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_prediction_examples(
    images: np.ndarray,
    test_idx: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    class_names: list[str],
    path: Path,
    count: int = 8,
) -> list[dict[str, object]]:
    selected = list(range(min(count, len(test_idx))))
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows))
    axes = np.asarray(axes).ravel()
    examples: list[dict[str, object]] = []
    for ax, local_index in zip(axes, selected):
        sample_idx = int(test_idx[local_index])
        true_label = int(y_test[local_index])
        predicted_label = int(pred[local_index])
        image = np.transpose(images[sample_idx], (1, 2, 0))
        image = image - image.min()
        image = image / max(float(image.max()), 1e-6)
        ax.imshow(image)
        ax.set_title(f"pred={class_names[predicted_label]}, true={class_names[true_label]}")
        ax.axis("off")
        examples.append(
            {
                "sample_index": sample_idx,
                "true_label": class_names[true_label],
                "predicted_label": class_names[predicted_label],
            }
        )
    for ax in axes[len(selected) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return examples


def write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "architecture",
        "description",
        "conv_layers",
        "filters",
        "kernel_sizes",
        "pool_after",
        "dropout",
        "activation",
        "fc_hidden",
        "seed",
        "val_accuracy",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "test_loss",
        "epochs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def architecture_table(configs: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Architektura | Conv vrstvy | Filtry | Kernel size | Pooling | Dropout | Aktivace | FFNN cast |",
        "| --- | ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for cfg in configs:
        lines.append(
            f"| `{cfg['name']}` | {len(cfg['channels'])} | {cfg['channels']} | {cfg['kernel_sizes']} | "
            f"po conv indexech {cfg['pool_after']} | {float(cfg['dropout']):.2f} | {cfg['activation']} | "
            f"Linear(feature_dim, {cfg['fc_hidden']}) -> {cfg['activation']} -> Linear({cfg['fc_hidden']}, {len(CLASS_NAMES)}) |"
        )
    return lines


def result_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1-score | Test loss | Epochs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['architecture']}` | {float(row['accuracy']):.4f} | {float(row['precision_macro']):.4f} | "
            f"{float(row['recall_macro']):.4f} | {float(row['f1_macro']):.4f} | "
            f"{float(row['test_loss']):.4f} | {int(row['epochs'])} |"
        )
    return lines


def write_report(
    report_path: Path,
    configs: list[dict[str, object]],
    rows: list[dict[str, object]],
    best: dict[str, object],
    dataset_info: dict[str, object],
    artifact_prefix: str,
) -> None:
    lines = [
        "# NNUI2 - Cviceni 9: Porovnani CNN architektur",
        "",
        "## 1. Cil experimentu",
        "Cilem je porovnat pet ruznych CNN architektur na stejne obrazove klasifikacni uloze a vyhodnotit vliv poctu konvolucnich vrstev, poolingu, dropoutu, aktivacni funkce a velikosti plne propojene casti.",
        "",
        f"Stav behu: `{RUN_STATUS}`. Experiment byl spusten jako kratky kontrolni beh, aby se overilo pouziti lokalniho flower datasetu a generovani vsech vystupu.",
        "",
        "## 2. Dataset",
        f"- Nazev datasetu: `{dataset_info['name']}`.",
        f"- Cesta k datasetu: `{dataset_info['path']}`.",
        f"- Pocet trid: `{dataset_info['classes']}`.",
        f"- Tridy: `{', '.join(dataset_info['class_names'])}`.",
        f"- Pocet obrazku: `{dataset_info['images']}`.",
        f"- Velikost obrazku po resize: `{dataset_info['image_size']}`.",
        f"- Preprocessing: `{dataset_info['preprocessing']}`.",
        f"- Train/val/test split: `{dataset_info['train_samples']}/{dataset_info['validation_samples']}/{dataset_info['test_samples']}` = `{dataset_info['split']}`.",
        "",
        "## 3. Spolecne trenovaci podminky",
        f"- Optimizer: `Adam`.",
        f"- Learning rate: `{LEARNING_RATE}`.",
        f"- Batch size: `{BATCH_SIZE}`.",
        f"- Epochs: maximalne `{MAX_EPOCHS}`, early stopping patience `{PATIENCE}` podle validation accuracy.",
        "- Loss function: `CrossEntropyLoss`.",
        f"- Seed: zakladni seed `{SEED}`, jednotlive architektury pouzivaji `SEED + index`.",
        f"- Device: `{DEVICE}`.",
        "",
        "## 4. Testovane architektury",
        *architecture_table(configs),
        "",
        "## 5. Vysledky",
        *result_table(rows),
        "",
        "## 6. Nejlepsi model",
        "Nejlepsi model byl vybran podle nejvyssi test accuracy; pri shode rozhoduje vyssi F1-score a nizsi test loss.",
        f"- Nejlepsi architektura: `{best['architecture']}`.",
        f"- Vysvetleni: tato varianta dosahla nejlepsi kombinace accuracy `{float(best['accuracy']):.4f}` a F1 `{float(best['f1_macro']):.4f}` pri test loss `{float(best['test_loss']):.4f}`.",
        f"- Ulozeny model: `{artifact_prefix}/best_model.pt`.",
        "",
        "## 7. Confusion matrix",
        f"Confusion matrix nejlepsiho modelu je ulozena v `{artifact_prefix}/best_confusion_matrix.png`.",
        "",
        f"![Confusion matrix]({artifact_prefix}/best_confusion_matrix.png)",
        "",
        "## 8. Train/validation loss graf",
        f"Train a validation loss krivky pro vsechny architektury jsou ulozeny v `{artifact_prefix}/loss_curves.png`.",
        "",
        f"![Loss curves]({artifact_prefix}/loss_curves.png)",
        "",
        "## 9. Ukazka predikci",
        f"Ukazky testovacich predikci nejlepsiho modelu jsou ulozeny v `{artifact_prefix}/prediction_examples.png`.",
        "",
        f"![Prediction examples]({artifact_prefix}/prediction_examples.png)",
        "",
        "## 10. Diskuze",
        "Flower dataset je narocnejsi nez male cislove datasety, protoze obsahuje RGB fotografie s vetsimi rozdily v pozadi, meritku, osvetleni a tvaru kvetu. Hlubsi CNN muze zachytit slozitejsi vizualni rysy, ale pri kratkem treninku nemusi vyuzit celou kapacitu.",
        "Pooling zmensuje prostorove rozliseni a pomaha potlacit male posuny objektu. U kvetin je to uzitecne, ale prilis agresivni zmenseni muze odstranit jemne textury okvetnich listku.",
        "Dropout pusobi jako regularizace plne propojene casti. U fotografii muze snizit preuceni na konkretni pozadi, ale pri nizkem poctu epoch muze take zpomalit uceni.",
        "Aktivacni funkce `ReLU` je rychla a stabilni pro vetsinu variant. `Tanh` v A5 meni nelinearitu a muze byt citlivejsi na saturaci, proto je vhodne sledovat nejen accuracy, ale i validation loss.",
        "Tento beh je kratky kontrolni beh nad realnym lokalnim flower datasetem. Vysledky jsou pouzitelne pro porovnani architektur za stejnych podminek, ale pro finalni presnost by bylo vhodne navysit pocet epoch a pripadne pouzit augmentaci.",
        "",
        "## 11. Zaver",
        "EXP09 splnuje zadani s omezenim kratkeho treninku: pouziva lokalni flower classification dataset z `/Users/dasky/PycharmProjects/cviceni/dataset/`, pet skutecne odlisnych PyTorch CNN architektur, stejne rozdeleni dat i trenovaci parametry, CSV metriky, porovnavaci graf, confusion matrix, loss krivky, ukazky predikci a ulozeny nejlepsi model.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = Path(__file__).resolve().parent
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)
    images, train_idx, val_idx, test_idx, y_train, y_val, y_test, class_names, dataset_info = prepare_data()
    rows: list[dict[str, object]] = []
    curves_by_arch: dict[str, dict[str, list[float]]] = {}
    best: dict[str, object] | None = None
    best_model: FlexibleCNN | None = None
    best_pred: np.ndarray | None = None
    best_config: dict[str, object] | None = None

    for index, cfg in enumerate(ARCHITECTURES, start=1):
        row, model, pred, curves = train_and_score(
            images,
            train_idx,
            val_idx,
            test_idx,
            y_train,
            y_val,
            y_test,
            cfg,
            seed=SEED + index,
            device=DEVICE,
        )
        rows.append(row)
        curves_by_arch[str(cfg["name"])] = curves
        torch.save({"config": cfg, "model_state": model.state_dict()}, results / f"{cfg['name']}.pt")
        candidate_key = (-float(row["accuracy"]), -float(row["f1_macro"]), float(row["test_loss"]))
        current_key = None
        if best is not None:
            current_key = (-float(best["accuracy"]), -float(best["f1_macro"]), float(best["test_loss"]))
        if best is None or candidate_key < current_key:
            best = row
            best_model = model
            best_pred = pred
            best_config = cfg

    if best is None or best_model is None or best_pred is None or best_config is None:
        raise RuntimeError("No EXP09 result was produced")

    write_metrics_csv(rows, results / "metrics.csv")
    (results / "results.json").write_text(
        json.dumps(
            {
                "framework": "pytorch",
                "status": RUN_STATUS,
                "dataset": dataset_info,
                "architectures": ARCHITECTURES,
                "rows": rows,
                "best": best,
                "curves": curves_by_arch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "config": best_config,
            "model_state": best_model.state_dict(),
            "metrics": best,
            "class_names": class_names,
        },
        results / "best_model.pt",
    )

    save_architecture_comparison(rows, results / "architecture_comparison.png")
    save_confusion(confusion_matrix(y_test, best_pred), class_names, results / "best_confusion_matrix.png")
    save_loss_curves(curves_by_arch, results / "loss_curves.png")
    examples = save_prediction_examples(images, test_idx, y_test, best_pred, class_names, results / "prediction_examples.png")
    (results / "prediction_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    write_report(ROOT / "experiment_09.md", ARCHITECTURES, rows, best, dataset_info, "EXP09/results")
    write_report(base / "report" / "report.md", ARCHITECTURES, rows, best, dataset_info, "../results")

    print("EXP09 completed")
    print(f"Metrics: {results / 'metrics.csv'}")
    print(f"Best model: {results / 'best_model.pt'}")
    print(f"Report: {ROOT / 'experiment_09.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
