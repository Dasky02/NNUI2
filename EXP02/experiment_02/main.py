from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime import configure_matplotlib_env

configure_matplotlib_env("nnui2_exp02")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.01, epochs: int = 100, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-0.5, 0.5, size=input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x: float) -> int:
        return 1 if x >= 0 else 0

    def predict(self, inputs: np.ndarray) -> int:
        potential = float(np.dot(inputs, self.weights[1:]) + self.weights[0])
        return self.activation(potential)

    def train(self, training_data: np.ndarray, labels: np.ndarray) -> list[int]:
        errors_history: list[int] = []
        for _ in range(self.epochs):
            epoch_errors = 0
            for inputs, label in zip(training_data, labels):
                label_int = int(label)
                prediction = self.predict(inputs)
                error = label_int - prediction
                if error != 0:
                    epoch_errors += 1
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
            errors_history.append(epoch_errors)
        return errors_history

    def test(self, testing_data: np.ndarray, labels: np.ndarray) -> float:
        predictions = np.array([self.predict(row) for row in testing_data], dtype=int)
        return float(np.mean(predictions == labels))

    def save(self, file_name: str | Path) -> None:
        np.save(file_name, self.weights)

    def load(self, file_name: str | Path) -> None:
        self.weights = np.load(file_name)


def prepare_dataset(random_state: int = 42) -> dict[str, np.ndarray | list[str]]:
    dataset = load_breast_cancer()
    feature_names = ["mean radius", "mean texture", "mean perimeter", "mean area"]
    feature_indices = [int(np.where(dataset.feature_names == name)[0][0]) for name in feature_names]

    X = dataset.data[:, feature_indices]
    y = dataset.target.astype(int)
    class_names = [str(name) for name in dataset.target_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return {
        "X_train": X_train_std,
        "X_test": X_test_std,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_names": class_names,
    }


def save_run_summary(run_rows: list[dict[str, float | int]], csv_path: Path) -> None:
    fieldnames = ["run", "seed", "final_train_error", "train_accuracy", "test_accuracy"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)


def plot_training_runs(histories: list[list[int]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for index, history in enumerate(histories, start=1):
        plt.plot(history, linewidth=1.2, label=f"beh {index}")
    plt.title("Průběh trénovací chyby pro 10 běhů")
    plt.xlabel("Epocha")
    plt.ylabel("Počet chyb v epoše")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_final_error_boxplot(final_errors: list[int], output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.boxplot(final_errors, patch_artist=True)
    plt.title("Boxplot finálních trénovacích chyb")
    plt.ylabel("Počet chyb v poslední epoše")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matice záměn nejlepšího modelu")
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
    plt.savefig(output_path, dpi=160)
    plt.close()


def generate_report(
    report_path: Path,
    feature_names: list[str],
    class_names: list[str],
    learning_rate: float,
    epochs: int,
    best_run: dict[str, object],
    final_errors: list[int],
    metrics: dict[str, float],
) -> None:
    lines = [
        "# Deník – experiment s jednoduchým perceptronem",
        "",
        "## Název experimentu",
        "Jednoduchý perceptron pro binární klasifikaci datasetu Breast Cancer Wisconsin.",
        "",
        "## Cíl úlohy",
        "Implementovat jednoduchý perceptron, ověřit jeho funkčnost a porovnat 10 trénování se stejnými hyperparametry a různou inicializací vah.",
        "",
        "## Popis zadání",
        "Zadání požadovalo veřejný dataset upravený pro perceptron, uložení průběhu trénovací chyby a finálních vah pro 10 běhů, výběr nejlepšího modelu a jeho vyhodnocení na testovacích datech pomocí matice záměn.",
        "",
        "## Použitá data / úprava datasetu",
        "Byl použit vestavěný dataset `sklearn.datasets.load_breast_cancer`.",
        f"Pro perceptron byly vybrány 4 příznaky: `{feature_names}`.",
        "Data byla rozdělena stratifikovaně na trénovací a testovací část v poměru 80:20 a standardizována podle statistik trénovací množiny.",
        "",
        "## Postup řešení",
        f"- Počet běhů: `10`",
        f"- Počet epoch: `{epochs}`",
        f"- Learning rate: `{learning_rate}`",
        "- V každém běhu byly uloženy průběhy chyb a finální váhy.",
        "- Nejlepší model byl zvolen podle nejnižší finální trénovací chyby; při shodě rozhodla vyšší testovací přesnost.",
        "",
        "## Implementace / zvolená metoda",
        "Perceptron používá binární aktivační funkci, predikci `sign(w·x + b)` a klasické perceptronové chybové učení.",
        "Implementovány jsou metody `activation`, `predict`, `train`, `test`, `save` a `load`.",
        "",
        "## Výsledky",
        f"- Nejlepší běh: `{best_run['run']}` se seedem `{best_run['seed']}`",
        f"- Finální trénovací chyba nejlepšího běhu: `{best_run['final_train_error']}`",
        f"- Test accuracy: `{metrics['accuracy']:.4f}`",
        f"- Precision: `{metrics['precision']:.4f}`",
        f"- Recall: `{metrics['recall']:.4f}`",
        f"- F1-score: `{metrics['f1']:.4f}`",
        f"- Rozptyl finálních chyb v 10 bězích: min `{min(final_errors)}`, medián `{float(np.median(final_errors)):.1f}`, max `{max(final_errors)}`",
        "",
        "## Vizualizace výsledků",
        "### Průběh trénovací chyby pro všech 10 běhů",
        "![training_error_runs](assets/training_error_runs.png)",
        "",
        "### Boxplot finálních trénovacích chyb",
        "![final_error_boxplot](assets/final_error_boxplot.png)",
        "",
        "### Matice záměn nejlepšího modelu",
        "![confusion_matrix](assets/confusion_matrix.png)",
        "",
        "## Diskuze výsledků",
        "Perceptron na vybraných čtyřech numerických příznacích dosahuje stabilně vysoké úspěšnosti, ale mezi běhy je patrný vliv počáteční inicializace vah na rychlost a přesnost konvergence.",
        "Rozdíly ve finální trénovací chybě potvrzují, že i jednoduchý lineární model může skončit v odlišných řešeních v závislosti na pořadí aktualizací a inicializaci.",
        "",
        "## Závěr",
        f"Experiment splnil zadání: implementace perceptronu byla ověřena na veřejném datasetu, byly vygenerovány všechny povinné výstupy a nejlepší model klasifikuje třídy `{class_names[0]}` a `{class_names[1]}` s testovací přesností `{metrics['accuracy']:.2%}`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"
    report_dir = base_dir / "report"
    assets_dir = report_dir / "assets"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(random_state=42)
    X_train = np.asarray(dataset["X_train"], dtype=float)
    X_test = np.asarray(dataset["X_test"], dtype=float)
    y_train = np.asarray(dataset["y_train"], dtype=int)
    y_test = np.asarray(dataset["y_test"], dtype=int)
    feature_names = list(dataset["feature_names"])
    class_names = list(dataset["class_names"])

    learning_rate = 0.01
    epochs = 100
    runs = 10

    histories: list[list[int]] = []
    final_errors: list[int] = []
    run_rows: list[dict[str, float | int]] = []
    models: list[tuple[dict[str, float | int], Perceptron]] = []

    for run in range(1, runs + 1):
        seed = 100 + run
        model = Perceptron(input_size=X_train.shape[1], learning_rate=learning_rate, epochs=epochs, seed=seed)
        history = model.train(X_train, y_train)
        predictions_train = np.array([model.predict(row) for row in X_train], dtype=int)
        predictions_test = np.array([model.predict(row) for row in X_test], dtype=int)

        row = {
            "run": run,
            "seed": seed,
            "final_train_error": int(history[-1]),
            "train_accuracy": float(accuracy_score(y_train, predictions_train)),
            "test_accuracy": float(accuracy_score(y_test, predictions_test)),
        }

        histories.append(history)
        final_errors.append(int(history[-1]))
        run_rows.append(row)
        models.append((row, model))

        model.save(outputs_dir / f"weights_run_{run}.npy")
        np.save(outputs_dir / f"errors_run_{run}.npy", np.asarray(history, dtype=int))

    save_run_summary(run_rows, outputs_dir / "run_summary.csv")

    best_row, best_model = min(
        models,
        key=lambda item: (int(item[0]["final_train_error"]), -float(item[0]["test_accuracy"])),
    )
    best_model.save(outputs_dir / "best_model_weights.npy")

    y_pred = np.array([best_model.predict(row) for row in X_test], dtype=int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    (outputs_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_run": best_row,
                "metrics": metrics,
                "confusion_matrix": cm.tolist(),
                "classification_report": report_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_training_runs(histories, assets_dir / "training_error_runs.png")
    plot_final_error_boxplot(final_errors, assets_dir / "final_error_boxplot.png")
    plot_confusion_matrix(cm, class_names, assets_dir / "confusion_matrix.png")

    generate_report(
        report_path=report_dir / "report.md",
        feature_names=feature_names,
        class_names=class_names,
        learning_rate=learning_rate,
        epochs=epochs,
        best_run=best_row,
        final_errors=final_errors,
        metrics=metrics,
    )

    print("EXP02 completed")
    print(f"Best run: {best_row['run']} (seed {best_row['seed']})")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Report: {report_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
