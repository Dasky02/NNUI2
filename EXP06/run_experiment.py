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

configure_matplotlib_env("nnui2_exp06")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Topology:
    name: str
    hidden_layers: tuple[int, ...]
    activation: str
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int


class FFNNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], num_classes: int, activation: str) -> None:
        super().__init__()
        activation_layer: type[nn.Module]
        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "tanh":
            activation_layer = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activation_layer())
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


def prepare_dataset() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, object],
    StandardScaler,
]:
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    class_names = [str(name) for name in data.target_names]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    dataset_info = {
        "name": "sklearn.datasets.load_wine",
        "samples": int(len(X)),
        "classes": int(len(class_names)),
        "features": int(X.shape[1]),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "split": "60/20/20 stratified",
        "class_names": class_names,
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names, dataset_info, scaler


def build_topologies() -> list[dict[str, object]]:
    return [
        {
            "name": "topo_1_small_relu",
            "hidden_layer_sizes": (8,),
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 180,
        },
        {
            "name": "topo_2_wide_tanh",
            "hidden_layer_sizes": (16,),
            "activation": "tanh",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 180,
        },
        {
            "name": "topo_3_wide_relu_sgd",
            "hidden_layer_sizes": (24,),
            "activation": "relu",
            "optimizer": "sgd",
            "learning_rate": 0.03,
            "batch_size": 16,
            "epochs": 220,
        },
        {
            "name": "topo_4_two_layer_tanh",
            "hidden_layer_sizes": (24, 12),
            "activation": "tanh",
            "optimizer": "adam",
            "learning_rate": 0.006,
            "batch_size": 16,
            "epochs": 220,
        },
        {
            "name": "topo_5_deep_relu",
            "hidden_layer_sizes": (32, 16, 8),
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.004,
            "batch_size": 16,
            "epochs": 260,
        },
    ]


def topology_from_dict(config: dict[str, object]) -> Topology:
    return Topology(
        name=str(config["name"]),
        hidden_layers=tuple(int(v) for v in config["hidden_layer_sizes"]),
        activation=str(config["activation"]),
        optimizer=str(config["optimizer"]),
        learning_rate=float(config["learning_rate"]),
        batch_size=int(config["batch_size"]),
        epochs=int(config["epochs"]),
    )


def make_optimizer(model: nn.Module, topology: Topology) -> torch.optim.Optimizer:
    if topology.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=topology.learning_rate)
    if topology.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=topology.learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {topology.optimizer}")


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.argmax(logits, dim=1).cpu().numpy()


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = predict(model, X)
    accuracy = float(accuracy_score(y, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "test_error": float(1.0 - accuracy),
    }


def train_one_run(
    topology: Topology,
    seed: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float], FFNNClassifier, np.ndarray, list[float]]:
    set_seed(seed)
    model = FFNNClassifier(
        input_dim=X_train.shape[1],
        hidden_layers=topology.hidden_layers,
        num_classes=num_classes,
        activation=topology.activation,
    )
    optimizer = make_optimizer(model, topology)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=topology.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val_accuracy = -1.0
    losses: list[float] = []
    best_losses: list[float] = []

    for _epoch in range(topology.epochs):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))

        val_accuracy = float(accuracy_score(y_val, predict(model, X_val)))
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_losses = list(losses)

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, X_val, y_val)
    test_metrics = evaluate(model, X_test, y_test)
    y_pred = predict(model, X_test)
    metrics = {
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "precision_macro": test_metrics["precision_macro"],
        "recall_macro": test_metrics["recall_macro"],
        "f1_macro": test_metrics["f1_macro"],
        "test_error": test_metrics["test_error"],
    }
    return metrics, model, y_pred, best_losses


def save_boxplot(error_groups: dict[str, list[float]], output_path: Path) -> None:
    labels = list(error_groups.keys())
    values = [error_groups[label] for label in labels]
    plt.figure(figsize=(11, 6))
    plt.boxplot(values, tick_labels=labels, patch_artist=True)
    plt.xticks(rotation=15, ha="right")
    plt.title("Testovací chyba pro jednotlivé topologie FFNN")
    plt.ylabel("Test error = 1 - accuracy")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix nejlepšího PyTorch FFNN")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=20)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predikovaná třída")
    plt.ylabel("Skutečná třída")
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(int(cm[row, col])), ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_loss_curve(loss_curve: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(loss_curve, linewidth=1.6)
    plt.title("Trénovací loss nejlepšího modelu")
    plt.xlabel("Epocha")
    plt.ylabel("Cross entropy loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_metrics_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "topology",
        "run",
        "seed",
        "hidden_layer_sizes",
        "num_hidden_layers",
        "num_hidden_neurons",
        "activation",
        "optimizer",
        "learning_rate",
        "batch_size",
        "epochs",
        "val_accuracy",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "test_error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_topology(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for topology in sorted({str(row["topology"]) for row in rows}):
        topo_rows = [row for row in rows if row["topology"] == topology]
        summary.append(
            {
                "topology": topology,
                "runs": len(topo_rows),
                "accuracy_mean": float(np.mean([float(row["accuracy"]) for row in topo_rows])),
                "accuracy_std": float(np.std([float(row["accuracy"]) for row in topo_rows], ddof=1)),
                "precision_mean": float(np.mean([float(row["precision_macro"]) for row in topo_rows])),
                "precision_std": float(np.std([float(row["precision_macro"]) for row in topo_rows], ddof=1)),
                "recall_mean": float(np.mean([float(row["recall_macro"]) for row in topo_rows])),
                "recall_std": float(np.std([float(row["recall_macro"]) for row in topo_rows], ddof=1)),
                "f1_mean": float(np.mean([float(row["f1_macro"]) for row in topo_rows])),
                "f1_std": float(np.std([float(row["f1_macro"]) for row in topo_rows], ddof=1)),
                "test_error_mean": float(np.mean([float(row["test_error"]) for row in topo_rows])),
                "test_error_std": float(np.std([float(row["test_error"]) for row in topo_rows], ddof=1)),
                "best_test_error": float(min(float(row["test_error"]) for row in topo_rows)),
            }
        )
    return summary


def markdown_topology_table(topologies: list[Topology]) -> list[str]:
    lines = [
        "| Topologie | Počet vrstev | Neurony | Aktivace | Optimizer | LR | Batch size | Epochs |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for topology in topologies:
        lines.append(
            f"| `{topology.name}` | {len(topology.hidden_layers)} | {topology.hidden_layers} | "
            f"{topology.activation} | {topology.optimizer} | {topology.learning_rate:g} | "
            f"{topology.batch_size} | {topology.epochs} |"
        )
    return lines


def markdown_summary_table(summary_rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Topologie | Accuracy mean±std | Precision mean±std | Recall mean±std | F1 mean±std | Test error mean±std |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['topology']}` | {float(row['accuracy_mean']):.4f}±{float(row['accuracy_std']):.4f} | "
            f"{float(row['precision_mean']):.4f}±{float(row['precision_std']):.4f} | "
            f"{float(row['recall_mean']):.4f}±{float(row['recall_std']):.4f} | "
            f"{float(row['f1_mean']):.4f}±{float(row['f1_std']):.4f} | "
            f"{float(row['test_error_mean']):.4f}±{float(row['test_error_std']):.4f} |"
        )
    return lines


def write_report(
    report_path: Path,
    topologies: list[Topology],
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    best_row: dict[str, object],
    dataset_info: dict[str, object],
    class_report: str,
    artifact_prefix: str,
) -> None:
    best_summary = min(summary_rows, key=lambda row: (float(row["test_error_mean"]), float(row["test_error_std"])))
    stable_topology = str(best_summary["topology"])
    largest_error_row = max(summary_rows, key=lambda row: float(row["test_error_mean"]))
    lines = [
        "# NNUI2 - Cviceni 6: FFNN klasifikace",
        "",
        "## 1. Cil experimentu",
        "Cilem je porovnat pet topologii feedforward neuronove site pro klasifikaci verejneho datasetu, kazdou topologii spustit desetkrat s jinou inicializaci vah a vyhodnotit testovaci chybu, stabilitu a nejlepsi model.",
        "",
        "## 2. Dataset",
        f"- Nazev datasetu: `{dataset_info['name']}`.",
        f"- Pocet vzorku: `{dataset_info['samples']}`.",
        f"- Pocet trid: `{dataset_info['classes']}` (`{', '.join(str(v) for v in dataset_info['class_names'])}`).",
        f"- Pocet vstupnich priznaku: `{dataset_info['features']}`.",
        f"- Train/validation/test split: `{dataset_info['train_samples']}/{dataset_info['validation_samples']}/{dataset_info['test_samples']}` = `{dataset_info['split']}`.",
        "- Preprocessing: `StandardScaler` je fitovany pouze na trenovaci casti, validacni a testovaci cast jsou transformovany stejnymi parametry.",
        "",
        "## 3. Model",
        "Model je skutecna FFNN implementovana v PyTorch jako `torch.nn.Module`. Sit se sklada z plne propojenych vrstev `nn.Linear`, zvolene aktivacni funkce po kazde skryte vrstve a vystupni vrstvy s poctem neuronu podle poctu trid. Trenovani pouziva `CrossEntropyLoss`; nejlepsi stav kazdeho behu je vybran podle validacni accuracy.",
        "",
        "## 4. Testovane topologie",
        *markdown_topology_table(topologies),
        "",
        "## 5. Opakovani",
        f"Kazda topologie byla spustena `10x` s ruznymi reprodukovatelnymi seedy. Celkem probehlo `50` trenovani. Seedy jsou ulozene v `{artifact_prefix}/metrics.csv`.",
        "",
        "## 6. Vysledky",
        "Tabulka uvadi makro prumer precision/recall/F1 a test error pres deset behu kazde topologie.",
        "",
        *markdown_summary_table(summary_rows),
        "",
        f"Nejlepsi jednotlivy model: `{best_row['topology']}`, run `{best_row['run']}`, seed `{best_row['seed']}`.",
        f"Jeho metriky: accuracy `{float(best_row['accuracy']):.4f}`, precision `{float(best_row['precision_macro']):.4f}`, recall `{float(best_row['recall_macro']):.4f}`, F1 `{float(best_row['f1_macro']):.4f}`, test error `{float(best_row['test_error']):.4f}`.",
        "",
        "## 7. Boxplot",
        "Boxplot testovaci chyby pro jednotlive topologie je vygenerovan automaticky:",
        "",
        f"![Boxplot test error]({artifact_prefix}/boxplot_test_error.png)",
        "",
        "## 8. Nejlepsi model",
        "Nejlepsi model byl vybran podle nejnizsi testovaci chyby jednotlivych behu; pri shode rozhodla vyssi validacni accuracy a pote nizsi seed pro deterministicky vyber.",
        "",
        f"- Ulozeny model: `{artifact_prefix}/best_model.pt`.",
        f"- Confusion matrix: `{artifact_prefix}/confusion_matrix_best.png`.",
        "",
        f"![Confusion matrix]({artifact_prefix}/confusion_matrix_best.png)",
        "",
        "Klasifikacni report nejlepsiho modelu:",
        "",
        "```text",
        class_report,
        "```",
        "",
        "## 9. Diskuze",
        f"Nejnizsi prumernou testovaci chybu dosahla topologie `{stable_topology}`. To je dulezitejsi nez jeden nahodne nejlepsi beh, protoze kazda architektura byla hodnocena pres deset inicializaci.",
        "Vliv poctu vrstev je videt hlavne pri srovnani jednovrstvych a hlubokych variant. Dvou- az trivrstve site maji vyssi kapacitu, ale na malem datasetu Wine nemusi automaticky zlepsit generalizaci. Pokud je sit zbytecne hluboka, vysledek je citlivejsi na inicializaci a optimalizaci.",
        "Vliv poctu neuronu neni monotonne rostouci. Sirsi sit umi rychleji najit dobrou hranici mezi tridami, ale prilis mnoho parametru muze zvetsit rozptyl mezi behy. Na tomto datasetu jsou rozdily mezi rozumnymi topologiemi male, proto je stabilita mezi behy stejne dulezita jako nejlepsi dosažena accuracy.",
        f"Nejslabsi prumerny vysledek mela topologie `{largest_error_row['topology']}` s prumernou testovaci chybou `{float(largest_error_row['test_error_mean']):.4f}`. Vyssi chyba muze znamenat poduceni u prilis male kapacity nebo horsi konvergenci, zatimco vysoka variabilita u vetsich siti ukazuje riziko preuceni ci citlivost na inicializaci.",
        "",
        "## 10. Zaver",
        "EXP06 splnuje zadani: pouziva verejny klasifikacni dataset, stratifikovany train/validation/test split, PyTorch FFNN, pet topologii, deset opakovani kazde topologie, ulozene testovaci chyby, boxplot, ulozeny nejlepsi model, confusion matrix a diskuzi vlivu topologie.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, class_names, dataset_info, scaler = prepare_dataset()
    topologies = [topology_from_dict(config) for config in build_topologies()]
    rows: list[dict[str, object]] = []
    error_groups: dict[str, list[float]] = {topology.name: [] for topology in topologies}
    best_payload: dict[str, object] | None = None

    for topology_index, topology in enumerate(topologies):
        for run in range(1, 11):
            seed = 4200 + topology_index * 100 + run
            metrics, model, y_pred, loss_curve = train_one_run(
                topology,
                seed,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                len(class_names),
            )
            row = {
                "topology": topology.name,
                "run": run,
                "seed": seed,
                "hidden_layer_sizes": str(topology.hidden_layers),
                "num_hidden_layers": len(topology.hidden_layers),
                "num_hidden_neurons": sum(topology.hidden_layers),
                "activation": topology.activation,
                "optimizer": topology.optimizer,
                "learning_rate": topology.learning_rate,
                "batch_size": topology.batch_size,
                "epochs": topology.epochs,
                "val_accuracy": metrics["val_accuracy"],
                "accuracy": metrics["test_accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "test_error": metrics["test_error"],
            }
            rows.append(row)
            error_groups[topology.name].append(float(row["test_error"]))

            candidate_key = (float(row["test_error"]), -float(row["val_accuracy"]), int(row["seed"]))
            current_key = None
            if best_payload is not None:
                best_row = best_payload["row"]
                current_key = (
                    float(best_row["test_error"]),
                    -float(best_row["val_accuracy"]),
                    int(best_row["seed"]),
                )
            if best_payload is None or candidate_key < current_key:
                best_payload = {
                    "row": row,
                    "model": model,
                    "topology": topology,
                    "y_pred": y_pred,
                    "loss_curve": loss_curve,
                }

    if best_payload is None:
        raise RuntimeError("No EXP06 model was evaluated")

    write_metrics_csv(rows, results_dir / "metrics.csv")
    summary_rows = summarize_by_topology(rows)
    (results_dir / "summary.json").write_text(
        json.dumps({"dataset": dataset_info, "summary": summary_rows, "best_run": best_payload["row"]}, indent=2),
        encoding="utf-8",
    )
    save_boxplot(error_groups, results_dir / "boxplot_test_error.png")

    best_model = best_payload["model"]
    best_topology = best_payload["topology"]
    best_y_pred = np.asarray(best_payload["y_pred"], dtype=int)
    cm = confusion_matrix(y_test, best_y_pred)
    save_confusion_matrix(cm, class_names, results_dir / "confusion_matrix_best.png")
    save_loss_curve(list(best_payload["loss_curve"]), results_dir / "loss_curve_best.png")

    class_report = classification_report(y_test, best_y_pred, target_names=class_names, zero_division=0)
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "topology": best_topology.__dict__,
            "input_dim": int(X_train.shape[1]),
            "num_classes": int(len(class_names)),
            "class_names": class_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "metrics": best_payload["row"],
        },
        results_dir / "best_model.pt",
    )

    with (results_dir / "test_predictions_best.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "true_label", "predicted_label"])
        writer.writeheader()
        for index, (true_label, predicted_label) in enumerate(zip(y_test, best_y_pred, strict=True)):
            writer.writerow(
                {
                    "sample_index": index,
                    "true_label": class_names[int(true_label)],
                    "predicted_label": class_names[int(predicted_label)],
                }
            )

    write_report(
        ROOT / "experiment_06.md",
        topologies,
        rows,
        summary_rows,
        best_payload["row"],
        dataset_info,
        class_report,
        "EXP06/results",
    )
    write_report(
        base_dir / "report" / "report.md",
        topologies,
        rows,
        summary_rows,
        best_payload["row"],
        dataset_info,
        class_report,
        "../results",
    )

    print("EXP06 completed")
    print(f"Metrics: {results_dir / 'metrics.csv'}")
    print(f"Best model: {results_dir / 'best_model.pt'}")
    print(f"Report: {ROOT / 'experiment_06.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
