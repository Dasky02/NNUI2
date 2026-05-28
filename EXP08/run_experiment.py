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

configure_matplotlib_env("nnui2_exp08")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


KERNEL_SIZES = [1, 3, 5, 7, 9]
SEED = 8080
LEARNING_RATE = 0.003
BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KernelCNN(nn.Module):
    def __init__(self, kernel_size: int, num_classes: int = 10) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(1, 8, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.features = nn.Sequential(self.conv, self.activation, self.pool)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 8, 8)
            feature_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    digits = load_digits()
    images = (digits.images.astype(np.float32) / 16.0)[:, None, :, :]
    labels = digits.target.astype(np.int64)
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        np.arange(len(labels)),
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )
    dataset_info = {
        "name": "sklearn.datasets.load_digits",
        "classes": 10,
        "images": int(len(images)),
        "image_size": "1x8x8 grayscale",
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "split": "60/20/20 stratified",
        "preprocessing": "pixel values scaled from 0-16 to 0-1; channel dimension added",
    }
    return images, train_idx, val_idx, test_idx, y_train, y_val, y_test, [str(i) for i in range(10)], dataset_info


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
    kernel_size: int,
    seed: int,
    device: torch.device = DEVICE,
) -> tuple[dict[str, object], KernelCNN, np.ndarray, list[float]]:
    set_seed(seed)
    model = KernelCNN(kernel_size=kernel_size).to(device)
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
    loss_curve: list[float] = []

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
        loss_curve.append(epoch_loss / max(seen, 1))

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
        "kernel_size": kernel_size,
        "padding": kernel_size // 2,
        "filters": 8,
        "seed": seed,
        "val_accuracy": float(accuracy_score(y_val, predict(model, images[val_idx], device))),
        "accuracy": test_accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "test_loss": evaluate_loss(model, images[test_idx], y_test, criterion, device),
        "epochs": len(loss_curve),
    }
    return row, model.cpu(), pred, loss_curve


def save_feature_maps(images: np.ndarray, models_by_kernel: dict[int, KernelCNN], test_idx: np.ndarray, path: Path) -> None:
    sample = int(test_idx[0])
    kernels = list(models_by_kernel)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.ravel()
    axes[0].imshow(images[sample, 0], cmap="gray")
    axes[0].set_title("vstup")
    x = torch.tensor(images[[sample]], dtype=torch.float32)
    for ax, kernel in zip(axes[1:], kernels):
        model = models_by_kernel[kernel]
        model.eval()
        with torch.no_grad():
            fmap = model.activation(model.conv(x))[0, 0].cpu().numpy()
        ax.imshow(fmap, cmap="magma")
        ax.set_title(f"kernel {kernel}x{kernel}")
    for ax in axes:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predikce")
    plt.ylabel("Skutecnost")
    plt.title("Confusion matrix nejlepsi CNN varianty kernelu")
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(int(cm[row, col])), ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_loss_curve(loss_curve: list[float], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(loss_curve)
    plt.xlabel("Epocha")
    plt.ylabel("Train loss")
    plt.title("Loss curve nejlepsi CNN varianty")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_kernel_barplot(rows: list[dict[str, object]], path: Path) -> None:
    kernels = [int(row["kernel_size"]) for row in rows]
    errors = [1.0 - float(row["accuracy"]) for row in rows]
    plt.figure(figsize=(8, 5))
    plt.bar([str(kernel) for kernel in kernels], errors, color="#4C78A8")
    plt.xlabel("Kernel size")
    plt.ylabel("Test error = 1 - accuracy")
    plt.title("Vliv velikosti kernelu na testovaci chybu")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "kernel_size",
        "padding",
        "filters",
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


def write_report(
    report_path: Path,
    rows: list[dict[str, object]],
    best: dict[str, object],
    dataset_info: dict[str, object],
    artifact_prefix: str,
) -> None:
    result_table = [
        "| Kernel | Accuracy | Precision | Recall | F1-score | Test loss | Epochs |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    kernel_table = [
        "| Varianta | Kernel size | Padding | Pocet filtru |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        result_table.append(
            f"| {int(row['kernel_size'])}x{int(row['kernel_size'])} | {float(row['accuracy']):.4f} | "
            f"{float(row['precision_macro']):.4f} | {float(row['recall_macro']):.4f} | "
            f"{float(row['f1_macro']):.4f} | {float(row['test_loss']):.4f} | {int(row['epochs'])} |"
        )
        kernel_table.append(
            f"| `kernel_{int(row['kernel_size'])}` | {int(row['kernel_size'])}x{int(row['kernel_size'])} | "
            f"{int(row['padding'])} | {int(row['filters'])} |"
        )

    lines = [
        "# NNUI2 - Cviceni 8: CNN a velikost konvolucniho jadra",
        "",
        "## 1. Cil experimentu",
        "Cilem je porovnat vliv velikosti konvolucniho kernelu v jedne CNN architekture na klasifikaci obrazku cislic. Mezi variantami se meni pouze `kernel_size` a odpovidajici `padding`, aby rozmery feature map zustaly korektni.",
        "",
        "## 2. Dataset",
        f"- Nazev datasetu: `{dataset_info['name']}`.",
        f"- Pocet trid: `{dataset_info['classes']}`.",
        f"- Pocet obrazku: `{dataset_info['images']}`.",
        f"- Velikost obrazku: `{dataset_info['image_size']}`.",
        f"- Train/val/test split: `{dataset_info['train_samples']}/{dataset_info['validation_samples']}/{dataset_info['test_samples']}` = `{dataset_info['split']}`.",
        f"- Preprocessing: `{dataset_info['preprocessing']}`.",
        "",
        "## 3. Architektura CNN",
        "Architektura je stejna pro vsechny varianty: `Conv2d(1, 8, kernel_size=k, padding=k//2) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(feature_dim, 32) -> ReLU -> Linear(32, 10)`. Pocet filtru, pooling, fully-connected cast, loss i optimizer zustavaji stejne.",
        "",
        "## 4. Testovane varianty kernelu",
        *kernel_table,
        "",
        "## 5. Trenovaci parametry",
        f"- Optimizer: `Adam`.",
        f"- Learning rate: `{LEARNING_RATE}`.",
        f"- Batch size: `{BATCH_SIZE}`.",
        f"- Epochs: maximalne `{MAX_EPOCHS}`, early stopping patience `{PATIENCE}` podle validation accuracy.",
        "- Loss function: `CrossEntropyLoss`.",
        f"- Seed: zakladni seed `{SEED}`, jednotlive varianty pouzivaji `SEED + kernel_size`.",
        f"- Device: `{DEVICE}`.",
        "",
        "## 6. Vysledky",
        *result_table,
        "",
        "## 7. Nejlepsi model",
        "Nejlepsi model byl vybran podle nejvyssi test accuracy; pri shode rozhoduje vyssi F1-score a nizsi test loss.",
        f"- Nejlepsi kernel: `{int(best['kernel_size'])}x{int(best['kernel_size'])}`.",
        f"- Metriky: accuracy `{float(best['accuracy']):.4f}`, precision `{float(best['precision_macro']):.4f}`, recall `{float(best['recall_macro']):.4f}`, F1 `{float(best['f1_macro']):.4f}`, test loss `{float(best['test_loss']):.4f}`.",
        f"- Ulozeny model: `{artifact_prefix}/best_model.pt`.",
        "",
        "## 8. Confusion matrix",
        f"Confusion matrix nejlepsiho modelu je ulozena v `{artifact_prefix}/best_confusion_matrix.png`.",
        "",
        f"![Confusion matrix]({artifact_prefix}/best_confusion_matrix.png)",
        "",
        "## 9. Feature mapy",
        f"Feature mapy prvniho filtru pro vsech pet kernelu jsou ulozeny v `{artifact_prefix}/feature_maps.png`.",
        "",
        f"![Feature maps]({artifact_prefix}/feature_maps.png)",
        "",
        "## 10. Diskuze",
        "Kernel `1x1` nevidi prostorove okoli pixelu, proto funguje hlavne jako kanalova transformace nad jednim vstupnim kanalem a ma omezenou schopnost zachytit tahy cislic. Kernely `3x3` a `5x5` lepe zachycuji lokalni tvary, hrany a kratke useky tahu.",
        "Vetsi kernely `7x7` a `9x9` maji sirsi receptive field uz v prvni vrstve, ale na obrazech 8x8 mohou prilis rychle agregovat velkou cast obrazku. To muze pomoci potlacit sum, ale soucasne hrozi ztrata jemnych detailu mezi podobnymi cislicemi.",
        "Protoze je padding nastaven na `kernel_size // 2`, rozmery po konvoluci zustavaji srovnatelne a rozdily ve vysledcich lze interpretovat hlavne jako vliv velikosti kernelu. Ostatni podminky, vcetne splitu, optimizeru, learning rate, batch size a architektury klasifikatoru, zustaly stejne.",
        "",
        "## 11. Zaver",
        "EXP08 splnuje zadani: pouziva mensi verejny obrazovy dataset, jednu PyTorch CNN architekturu, pet variant kernelu `1x1`, `3x3`, `5x5`, `7x7`, `9x9`, stejne trenovaci podminky, CSV metriky, porovnavaci graf, confusion matrix, feature mapy a ulozeny nejlepsi model.",
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
    models_by_kernel: dict[int, KernelCNN] = {}
    best: dict[str, object] | None = None
    best_model: KernelCNN | None = None
    best_pred: np.ndarray | None = None
    best_loss: list[float] | None = None

    for kernel in KERNEL_SIZES:
        row, model, pred, loss_curve = train_and_score(
            images,
            train_idx,
            val_idx,
            test_idx,
            y_train,
            y_val,
            y_test,
            kernel,
            seed=SEED + kernel,
            device=DEVICE,
        )
        rows.append(row)
        models_by_kernel[kernel] = model
        torch.save({"kernel_size": kernel, "padding": kernel // 2, "model_state": model.state_dict()}, results / f"model_kernel_{kernel}.pt")
        candidate_key = (-float(row["accuracy"]), -float(row["f1_macro"]), float(row["test_loss"]))
        current_key = None
        if best is not None:
            current_key = (-float(best["accuracy"]), -float(best["f1_macro"]), float(best["test_loss"]))
        if best is None or candidate_key < current_key:
            best = row
            best_model = model
            best_pred = pred
            best_loss = loss_curve

    if best is None or best_model is None or best_pred is None or best_loss is None:
        raise RuntimeError("No EXP08 result was produced")

    write_metrics_csv(rows, results / "metrics.csv")
    write_metrics_csv(rows, results / "kernel_comparison.csv")
    (results / "results.json").write_text(
        json.dumps({"framework": "pytorch", "dataset": dataset_info, "rows": rows, "best": best}, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "kernel_size": best["kernel_size"],
            "padding": best["padding"],
            "filters": best["filters"],
            "model_state": best_model.state_dict(),
            "metrics": best,
            "class_names": class_names,
        },
        results / "best_model.pt",
    )
    save_feature_maps(images, models_by_kernel, test_idx, results / "feature_maps.png")
    save_confusion(confusion_matrix(y_test, best_pred), class_names, results / "best_confusion_matrix.png")
    save_loss_curve(best_loss, results / "best_loss_curve.png")
    save_kernel_barplot(rows, results / "kernel_boxplot_or_barplot.png")
    write_report(ROOT / "experiment_08.md", rows, best, dataset_info, "EXP08/results")
    write_report(base / "report" / "report.md", rows, best, dataset_info, "../results")

    print("EXP08 completed")
    print(f"Metrics: {results / 'metrics.csv'}")
    print(f"Best model: {results / 'best_model.pt'}")
    print(f"Report: {ROOT / 'experiment_08.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
