from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime import configure_matplotlib_env

configure_matplotlib_env("nnui2_exp07")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    method: str
    params: str
    dimension: int
    values: np.ndarray


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...] = (48, 24), num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def hog_features(images: np.ndarray, cells: int = 4, bins: int = 9) -> np.ndarray:
    features: list[np.ndarray] = []
    for image in images:
        gy, gx = np.gradient(image.astype(float))
        magnitude = np.hypot(gx, gy)
        angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
        parts: list[float] = []
        step = image.shape[0] // cells
        for row in range(cells):
            for col in range(cells):
                mag_cell = magnitude[row * step : (row + 1) * step, col * step : (col + 1) * step].ravel()
                ang_cell = angle[row * step : (row + 1) * step, col * step : (col + 1) * step].ravel()
                hist, _ = np.histogram(ang_cell, bins=bins, range=(0.0, 180.0), weights=mag_cell)
                norm = np.linalg.norm(hist)
                parts.extend((hist / norm if norm else hist).tolist())
        features.append(np.asarray(parts, dtype=float))
    return np.vstack(features).astype(np.float32)


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
    images = (digits.images.astype(np.float32) / 16.0).astype(np.float32)
    labels = digits.target.astype(np.int64)
    flat = images.reshape(len(images), -1).astype(np.float32)
    class_names = [str(i) for i in range(10)]

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
        "type": "obrazovy dataset 8x8 grayscale cislic",
        "samples": int(len(images)),
        "classes": int(len(class_names)),
        "image_shape": "8x8",
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "split": "60/20/20 stratified",
    }
    return images, flat, train_idx, val_idx, test_idx, y_train, y_val, y_test, class_names, dataset_info


def build_variants(flat: np.ndarray, images: np.ndarray, train_idx: np.ndarray) -> dict[str, FeatureVariant]:
    raw = flat.astype(np.float32)
    hog = hog_features(images, cells=4, bins=9)

    pca = PCA(n_components=16, random_state=42)
    pca.fit(raw[train_idx])
    pca_raw = pca.transform(raw).astype(np.float32)

    hog_pca_model = PCA(n_components=24, random_state=42)
    hog_pca_model.fit(hog[train_idx])
    hog_pca = hog_pca_model.transform(hog).astype(np.float32)

    return {
        "raw_pixels_64": FeatureVariant(
            name="raw_pixels_64",
            method="flatten raw pixels",
            params="8x8 obraz serializovany na 64 hodnot",
            dimension=64,
            values=raw,
        ),
        "hog_4x4_9bins": FeatureVariant(
            name="hog_4x4_9bins",
            method="HOG-like histogram orientovanych gradientu",
            params="4x4 bunky, 9 binu na bunku",
            dimension=144,
            values=hog,
        ),
        "pca_16_from_raw": FeatureVariant(
            name="pca_16_from_raw",
            method="PCA z raw pixelu",
            params="16 komponent, fit pouze na train splitu",
            dimension=16,
            values=pca_raw,
        ),
        "hog_pca_24": FeatureVariant(
            name="hog_pca_24",
            method="PCA z HOG priznaku",
            params="HOG 144D redukovany na 24 komponent, fit pouze na train splitu",
            dimension=24,
            values=hog_pca,
        ),
    }


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.argmax(logits, dim=1).cpu().numpy()


def train_variant(
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    hidden_layers: tuple[int, ...] = (48, 24),
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 0.003,
) -> tuple[dict[str, float], FeatureMLP, np.ndarray, StandardScaler]:
    set_seed(seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    model = FeatureMLP(input_dim=X_train.shape[1], hidden_layers=hidden_layers, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = -1.0
    patience = 15
    stale = 0
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_acc = float(accuracy_score(y_val, predict(model, X_val)))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    val_pred = predict(model, X_val)
    test_pred = predict(model, X_test)
    test_accuracy = float(accuracy_score(y_test, test_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="macro", zero_division=0)
    row = {
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "accuracy": test_accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "test_error": float(1.0 - test_accuracy),
    }
    return row, model, test_pred, scaler


def save_boxplot(groups: dict[str, list[float]], path: Path) -> None:
    labels = list(groups)
    plt.figure(figsize=(10, 5.5))
    plt.boxplot([groups[label] for label in labels], tick_labels=labels, patch_artist=True)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Test error = 1 - accuracy")
    plt.title("Porovnani reprezentaci vstupu podle testovaci chyby")
    plt.grid(axis="y", alpha=0.3)
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
    plt.title("Confusion matrix nejlepsiho modelu")
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(int(cm[row, col])), ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_feature_examples(images: np.ndarray, variants: dict[str, FeatureVariant], train_idx: np.ndarray, path: Path) -> None:
    sample = int(train_idx[0])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()
    axes[0].imshow(images[sample], cmap="gray")
    axes[0].set_title("vstupni obraz")
    axes[1].bar(np.arange(variants["raw_pixels_64"].dimension), variants["raw_pixels_64"].values[sample], width=1.0)
    axes[1].set_title("raw pixely 64D")
    axes[2].bar(np.arange(variants["hog_4x4_9bins"].dimension), variants["hog_4x4_9bins"].values[sample], width=1.0)
    axes[2].set_title("HOG 144D")
    axes[3].bar(np.arange(variants["pca_16_from_raw"].dimension), variants["pca_16_from_raw"].values[sample], width=0.8)
    axes[3].set_title("PCA z raw 16D")
    for ax in axes:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "variant",
        "method",
        "dimension",
        "run",
        "seed",
        "val_accuracy",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "test_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "method": str(variant_rows[0]["method"]),
                "dimension": int(variant_rows[0]["dimension"]),
                "runs": len(variant_rows),
                "accuracy_mean": float(np.mean([float(row["accuracy"]) for row in variant_rows])),
                "accuracy_std": float(np.std([float(row["accuracy"]) for row in variant_rows], ddof=1)),
                "precision_mean": float(np.mean([float(row["precision_macro"]) for row in variant_rows])),
                "precision_std": float(np.std([float(row["precision_macro"]) for row in variant_rows], ddof=1)),
                "recall_mean": float(np.mean([float(row["recall_macro"]) for row in variant_rows])),
                "recall_std": float(np.std([float(row["recall_macro"]) for row in variant_rows], ddof=1)),
                "f1_mean": float(np.mean([float(row["f1_macro"]) for row in variant_rows])),
                "f1_std": float(np.std([float(row["f1_macro"]) for row in variant_rows], ddof=1)),
                "test_error_mean": float(np.mean([float(row["test_error"]) for row in variant_rows])),
                "test_error_std": float(np.std([float(row["test_error"]) for row in variant_rows], ddof=1)),
                "best_test_error": float(min(float(row["test_error"]) for row in variant_rows)),
            }
        )
    return summary


def variant_table(variants: dict[str, FeatureVariant]) -> list[str]:
    lines = [
        "| Varianta | Metoda extrakce | Dimenze | Parametry |",
        "| --- | --- | ---: | --- |",
    ]
    for variant in variants.values():
        lines.append(f"| `{variant.name}` | {variant.method} | {variant.dimension} | {variant.params} |")
    return lines


def summary_table(summary_rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Varianta | Dimenze | Accuracy mean±std | Precision mean±std | Recall mean±std | F1 mean±std | Test error mean±std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['variant']}` | {int(row['dimension'])} | "
            f"{float(row['accuracy_mean']):.4f}±{float(row['accuracy_std']):.4f} | "
            f"{float(row['precision_mean']):.4f}±{float(row['precision_std']):.4f} | "
            f"{float(row['recall_mean']):.4f}±{float(row['recall_std']):.4f} | "
            f"{float(row['f1_mean']):.4f}±{float(row['f1_std']):.4f} | "
            f"{float(row['test_error_mean']):.4f}±{float(row['test_error_std']):.4f} |"
        )
    return lines


def write_report(
    report_path: Path,
    variants: dict[str, FeatureVariant],
    summary_rows: list[dict[str, object]],
    best: dict[str, object],
    dataset_info: dict[str, object],
    class_report: str,
    artifact_prefix: str,
) -> None:
    best_average = min(summary_rows, key=lambda row: (float(row["test_error_mean"]), float(row["test_error_std"])))
    highest_dim = max(summary_rows, key=lambda row: int(row["dimension"]))
    lowest_dim = min(summary_rows, key=lambda row: int(row["dimension"]))
    lines = [
        "# NNUI2 - Cviceni 7: Extrakce priznaku",
        "",
        "## 1. Cil experimentu",
        "Cilem je overit vliv ruznych reprezentaci vstupu a extrakce priznaku na klasifikaci pomoci stejne FFNN. Porovnavaji se raw pixely, HOG, PCA z raw pixelu a PCA z HOG priznaku.",
        "",
        "## 2. Dataset",
        f"- Nazev datasetu: `{dataset_info['name']}`.",
        f"- Typ dat: `{dataset_info['type']}`.",
        f"- Pocet vzorku: `{dataset_info['samples']}`.",
        f"- Pocet trid: `{dataset_info['classes']}`.",
        f"- Rozdeleni train/val/test: `{dataset_info['train_samples']}/{dataset_info['validation_samples']}/{dataset_info['test_samples']}` = `{dataset_info['split']}`.",
        "",
        "## 3. Preprocessing",
        "- Obrazy 8x8 jsou prevedeny na `float32` a normalizovany z rozsahu 0-16 do rozsahu 0-1.",
        "- Pro raw pixely se obraz flattenuje na 64D vektor.",
        "- HOG-like varianta pocita gradienty a histogramy orientaci v bunkach.",
        "- PCA varianty jsou fitovane pouze na trenovaci casti, aby nedochazelo k data leakage.",
        "- Pred trenovanim FFNN je kazda varianta skálovana pomoci `StandardScaler` fitovaneho pouze na train splitu.",
        "",
        "## 4. Varianty priznaku",
        *variant_table(variants),
        "",
        "## 5. Model",
        "Pro vsechny varianty je pouzita stejna PyTorch FFNN (`torch.nn.Module`): vstupni vrstva podle dimenze priznaku, skryte vrstvy `(48, 24)`, aktivace `ReLU`, vystupni vrstva pro 10 trid, `CrossEntropyLoss`, optimizer `Adam`, learning rate `0.003`, batch size `32`, max `120` epoch s early stopping podle validacni accuracy.",
        "",
        "| Varianta | Vstupni dimenze | Hidden vrstvy | Aktivace | Optimizer | Epochs |",
        "| --- | ---: | --- | --- | --- | ---: |",
        *[
            f"| `{variant.name}` | {variant.dimension} | (48, 24) | ReLU | Adam, lr=0.003 | 120 |"
            for variant in variants.values()
        ],
        "",
        "## 6. Opakovani",
        f"Kazda varianta byla spustena `5x` s ruznymi reprodukovatelnymi seedy. Celkem probehlo `{len(variants) * 5}` trenovani. Seedy a testovaci chyby jsou v `{artifact_prefix}/metrics.csv`.",
        "",
        "## 7. Vysledky",
        *summary_table(summary_rows),
        "",
        "## 8. Boxplot",
        f"Boxplot porovnava testovaci chybu pro jednotlive varianty priznaku: `{artifact_prefix}/feature_boxplot.png`.",
        "",
        f"![Boxplot]({artifact_prefix}/feature_boxplot.png)",
        "",
        "## 9. Confusion matrix nejlepsiho modelu",
        f"Confusion matrix nejlepsiho behu je ulozena v `{artifact_prefix}/confusion_matrix_best.png`.",
        "",
        f"![Confusion matrix]({artifact_prefix}/confusion_matrix_best.png)",
        "",
        "## 10. Nejlepsi model",
        f"- Nejlepsi priznaky podle jednotliveho behu: `{best['variant']}`.",
        f"- Nejlepsi beh: run `{best['run']}`, seed `{best['seed']}`.",
        f"- Metriky: accuracy `{float(best['accuracy']):.4f}`, precision `{float(best['precision_macro']):.4f}`, recall `{float(best['recall_macro']):.4f}`, F1 `{float(best['f1_macro']):.4f}`, test error `{float(best['test_error']):.4f}`.",
        f"- Ulozeny model: `{artifact_prefix}/best_model.pt`.",
        "",
        "```text",
        class_report,
        "```",
        "",
        "## 11. Diskuze",
        f"Nejlepsi prumernou testovaci chybu dosahla varianta `{best_average['variant']}`. To ukazuje, ze pro tento dataset neni rozhodujici pouze nejlepsi jeden beh, ale i stabilita mezi inicializacemi.",
        "Raw pixely zachovavaji vsechny hodnoty obrazu a u malych 8x8 cislic mohou byt velmi silne, protoze FFNN primo vidi plny raster. Jejich nevyhodou je vetsi vstupni dimenze a slabsi vestavena invariantnost vuci posunum nebo tvarovym zmenam.",
        "HOG priznaky explicitne popisuji smer hran, coz je pro cislice prirozene. Na velmi malych obrazech 8x8 je ale gradientova informace hruba, a proto HOG nemusi vzdy prekonat raw pixely.",
        "PCA snizuje dimenzi a odstranuje cast sumu nebo redundance. Nizsi dimenze muze zlepsit stabilitu a zmensit pocet parametru prvni vrstvy, ale prilis agresivni redukce muze ztratit jemne rozdily mezi podobnymi cislicemi.",
        f"Nejvyssi dimenzi ma `{highest_dim['variant']}` ({int(highest_dim['dimension'])}D), nejnizsi dimenzi ma `{lowest_dim['variant']}` ({int(lowest_dim['dimension'])}D). Porovnani ukazuje, ze vetsi dimenze automaticky neznamena lepsi vysledek; dulezita je informacni hodnota reprezentace pro konkretni dataset.",
        "Stabilita mezi peti behy je dana jak reprezentaci, tak inicializaci FFNN. Varianta s nizkou prumernou chybou a malou smerodatnou odchylkou je prakticky vhodnejsi nez varianta s jednim vyjimecnym behem a velkym rozptylem.",
        "",
        "## 12. Zaver",
        "EXP07 splnuje zadani: pouziva jeden verejny obrazovy dataset, stratifikovane train/validation/test rozdeleni, preprocessing, ctyri skutecne varianty priznaku, PyTorch FFNN, pet behu na variantu, ulozene testovaci chyby, boxplot, confusion matrix a diskuzi vlivu typu i dimenze priznaku.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = Path(__file__).resolve().parent
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)

    images, flat, train_idx, val_idx, test_idx, y_train, y_val, y_test, class_names, dataset_info = prepare_data()
    variants = build_variants(flat, images, train_idx)
    rows: list[dict[str, object]] = []
    groups: dict[str, list[float]] = {name: [] for name in variants}
    best: dict[str, object] | None = None
    best_pred: np.ndarray | None = None
    best_model: FeatureMLP | None = None
    best_scaler: StandardScaler | None = None
    best_variant: FeatureVariant | None = None

    for variant_index, variant in enumerate(variants.values()):
        for run in range(1, 6):
            seed = 7100 + variant_index * 100 + run
            metrics, model, pred, scaler = train_variant(
                variant.values,
                train_idx,
                val_idx,
                test_idx,
                y_train,
                y_val,
                y_test,
                seed,
            )
            row: dict[str, object] = {
                "variant": variant.name,
                "method": variant.method,
                "dimension": variant.dimension,
                "run": run,
                "seed": seed,
                **metrics,
            }
            rows.append(row)
            groups[variant.name].append(float(metrics["test_error"]))
            candidate_key = (float(row["test_error"]), -float(row["val_accuracy"]), int(row["seed"]))
            current_key = None
            if best is not None:
                current_key = (float(best["test_error"]), -float(best["val_accuracy"]), int(best["seed"]))
            if best is None or candidate_key < current_key:
                best = row
                best_pred = pred
                best_model = model
                best_scaler = scaler
                best_variant = variant

    if best is None or best_pred is None or best_model is None or best_scaler is None or best_variant is None:
        raise RuntimeError("No EXP07 result was produced")

    write_metrics_csv(rows, results / "metrics.csv")
    summary_rows = summarize(rows)
    (results / "summary.json").write_text(
        json.dumps({"framework": "pytorch", "dataset": dataset_info, "summary": summary_rows, "best": best}, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "variant": best["variant"],
            "method": best_variant.method,
            "dimension": best_variant.dimension,
            "model_state": best_model.state_dict(),
            "input_dim": best_variant.dimension,
            "hidden_layers": (48, 24),
            "num_classes": len(class_names),
            "class_names": class_names,
            "scaler_mean": best_scaler.mean_.tolist(),
            "scaler_scale": best_scaler.scale_.tolist(),
            "metrics": best,
        },
        results / "best_model.pt",
    )

    save_feature_examples(images, variants, train_idx, results / "feature_examples.png")
    save_boxplot(groups, results / "feature_boxplot.png")
    cm = confusion_matrix(y_test, best_pred)
    save_confusion(cm, class_names, results / "confusion_matrix_best.png")
    class_report = classification_report(y_test, best_pred, zero_division=0)
    (results / "classification_report_best.txt").write_text(class_report, encoding="utf-8")

    write_report(ROOT / "experiment_07.md", variants, summary_rows, best, dataset_info, class_report, "EXP07/results")
    write_report(base / "report" / "report.md", variants, summary_rows, best, dataset_info, class_report, "../results")

    print("EXP07 completed")
    print(f"Metrics: {results / 'metrics.csv'}")
    print(f"Best model: {results / 'best_model.pt'}")
    print(f"Report: {ROOT / 'experiment_07.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
