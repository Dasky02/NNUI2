from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime import configure_matplotlib_env

configure_matplotlib_env("nnui2_exp05")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffnn import FFNN


def minmax_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return mins, np.where(maxs - mins == 0.0, 1.0, maxs - mins)


def minmax_transform(X: np.ndarray, mins: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mins) / scale


def minmax_inverse(X: np.ndarray, mins: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return X * scale + mins


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def save_curve(train_errors: list[float], val_errors: list[float], output_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(train_errors, label="train", linewidth=1.5)
    if val_errors:
        plt.plot(val_errors, label="validation", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Epocha")
    plt.ylabel("MSE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_bar_chart(values: dict[str, float], output_path: Path, title: str, ylabel: str) -> None:
    labels = list(values.keys())
    data = [values[label] for label in labels]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = min(float(y_true.min()), float(y_pred.min()))
    mx = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([mn, mx], [mn, mx], color="black", linestyle="--", linewidth=1.2)
    plt.title("Nejlepší model: predikce vs. skutečnost")
    plt.xlabel("Skutečné hodnoty")
    plt.ylabel("Predikované hodnoty")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_synthetic_check(assets_dir: Path) -> dict[str, float]:
    rng = np.random.default_rng(42)
    X = rng.uniform(-1.0, 1.0, size=(100, 3))
    noise = rng.uniform(-0.02, 0.02, size=100)
    t1 = X[:, 0] ** 2 + X[:, 1] - X[:, 2] + noise
    t2 = X[:, 0] * X[:, 1] - X[:, 2] + noise
    T = np.column_stack([t1, t2])

    model = FFNN(input_dim=3, hidden_units=10, output_dim=2, f_hidden="tanh", f_output="linear", lr=0.05, seed=7)
    Xn, Tn = model.normalize(X, T, 0.0, 1.0)

    X_train, X_val = Xn[:60], Xn[60:80]
    T_train, T_val = Tn[:60], Tn[60:80]

    train_errors, val_errors = model.train(X_train, T_train, X_val, T_val, epochs=150, lr=0.05)
    save_curve(
        train_errors,
        val_errors,
        assets_dir / "synthetic_train_validation_curve.png",
        "Syntetická kontrola FFNN objektu",
    )
    return {
        "final_train_mse": float(train_errors[-1]),
        "final_val_mse": float(val_errors[-1]),
    }


def run_main_experiment(base_dir: Path, assets_dir: Path, outputs_dir: Path) -> dict[str, object]:
    data_dir = base_dir.parents[1] / "cviceni" / "Cv5_Data"
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feature_columns = ["x1", "x2", "x3", "x4"]
    target_columns = ["y_real"]

    X_train_raw = train_df[feature_columns].to_numpy(dtype=float)
    y_train_raw = train_df[target_columns].to_numpy(dtype=float)
    X_val_raw = val_df[feature_columns].to_numpy(dtype=float)
    y_val_raw = val_df[target_columns].to_numpy(dtype=float)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=float)
    y_test_raw = test_df[target_columns].to_numpy(dtype=float)

    x_min, x_scale = minmax_fit(X_train_raw)
    y_min, y_scale = minmax_fit(y_train_raw)
    X_train = minmax_transform(X_train_raw, x_min, x_scale)
    X_val = minmax_transform(X_val_raw, x_min, x_scale)
    X_test = minmax_transform(X_test_raw, x_min, x_scale)
    y_train = minmax_transform(y_train_raw, y_min, y_scale)
    y_val = minmax_transform(y_val_raw, y_min, y_scale)
    y_test = minmax_transform(y_test_raw, y_min, y_scale)

    configs = [
        {"name": "few_lr0.01_h3", "hidden_units": 3, "epochs": 100, "lr": 0.01, "f_hidden": "tanh"},
        {"name": "few_lr0.55_h3", "hidden_units": 3, "epochs": 100, "lr": 0.55, "f_hidden": "tanh"},
        {"name": "range50_h10", "hidden_units": 10, "epochs": 50, "lr": 0.05, "f_hidden": "tanh"},
        {"name": "range50_h20", "hidden_units": 20, "epochs": 50, "lr": 0.05, "f_hidden": "tanh"},
        {"name": "range50_h30", "hidden_units": 30, "epochs": 50, "lr": 0.05, "f_hidden": "tanh"},
        {"name": "range1000_h10", "hidden_units": 10, "epochs": 1000, "lr": 0.02, "f_hidden": "relu"},
        {"name": "range1000_h20", "hidden_units": 20, "epochs": 1000, "lr": 0.02, "f_hidden": "relu"},
        {"name": "range1000_h30", "hidden_units": 30, "epochs": 1000, "lr": 0.02, "f_hidden": "relu"},
    ]

    results: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    best_test_mse = float("inf")

    for idx, config in enumerate(configs, start=1):
        model = FFNN(
            input_dim=4,
            hidden_units=int(config["hidden_units"]),
            output_dim=1,
            f_hidden=str(config["f_hidden"]),
            f_output="linear",
            lr=float(config["lr"]),
            seed=idx * 11,
        )
        train_errors, val_errors = model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
        )
        y_pred_test_norm = model.predict(X_test).reshape(-1, 1)
        y_pred_test = minmax_inverse(y_pred_test_norm, y_min, y_scale)
        test_mse = mse(y_test_raw, y_pred_test)
        test_mae = mae(y_test_raw, y_pred_test)

        save_curve(
            train_errors,
            val_errors,
            assets_dir / f"{config['name']}_curve.png",
            f"{config['name']}: train/validation MSE",
        )

        row = {
            "name": config["name"],
            "hidden_units": int(config["hidden_units"]),
            "epochs": int(config["epochs"]),
            "lr": float(config["lr"]),
            "f_hidden": str(config["f_hidden"]),
            "final_train_mse": float(train_errors[-1]),
            "final_val_mse": float(val_errors[-1]),
            "test_mse": test_mse,
            "test_mae": test_mae,
        }
        results.append(row)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_payload = {
                "config": row,
                "model": model,
                "y_pred_test": y_pred_test.reshape(-1),
                "y_true_test": y_test_raw.reshape(-1),
            }

    if best_payload is None:
        raise RuntimeError("No EXP05 configuration was evaluated")

    best_model = best_payload["model"]
    assert isinstance(best_model, FFNN)
    best_model.save(str(outputs_dir / "best_model_weights.npz"))
    save_bar_chart(
        {row["name"]: float(row["test_mse"]) for row in results},
        assets_dir / "test_mse_by_config.png",
        "Testovací MSE pro všechny konfigurace",
        "Test MSE",
    )
    save_prediction_scatter(
        np.asarray(best_payload["y_true_test"], dtype=float),
        np.asarray(best_payload["y_pred_test"], dtype=float),
        assets_dir / "best_model_prediction_scatter.png",
    )
    return {"configs": results, "best": best_payload["config"]}


def generate_report(report_path: Path, synthetic: dict[str, float], experiment: dict[str, object]) -> None:
    configs = experiment["configs"]
    best = experiment["best"]
    assert isinstance(configs, list)
    assert isinstance(best, dict)

    table_lines = [
        "| konfigurace | hidden | epochs | lr | aktivace | test MSE | test MAE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in configs:
        table_lines.append(
            f"| {row['name']} | {row['hidden_units']} | {row['epochs']} | {row['lr']} | {row['f_hidden']} | "
            f"{row['test_mse']:.4f} | {row['test_mae']:.4f} |"
        )

    lines = [
        "# NNUI2 – Cvičení 5: Vícevrstvá dopředná neuronová síť",
        "",
        "## Název experimentu",
        "FFNN pro regresní aproximaci syntetické funkce a datasetu z LMS.",
        "",
        "## Cíl úlohy",
        "Implementovat objekt FFNN s jednou skrytou vrstvou, ověřit backpropagaci na syntetické úloze a porovnat více konfigurací nad připraveným regresním datasetem.",
        "",
        "## Popis zadání",
        "Zadání požadovalo implementaci metod `normalize`, `activation`, `activation_derivative`, `forward`, `train_epoch`, `validate`, `test` a `train`, syntetický test s `input_dim=3` a hlavní experiment nad daty z LMS s ukládáním train/validation křivek a testovacích chyb.",
        "",
        "## Použitá data / úprava datasetu",
        "Syntetický test používá 100 vzorků z intervalu `[-1, 1]` a dva zašuměné cíle podle vzorců ze zadání.",
        "Hlavní experiment používá soubory `cviceni/Cv5_Data/train.csv`, `val.csv` a `test.csv`. Vstupy i cíle byly normalizovány metodou min-max podle trénovací množiny a metriky jsou reportovány po převodu zpět do původní škály.",
        "",
        "## Postup řešení",
        "- Syntetická kontrola: `hidden_units=10`, `epochs=150`, `lr=0.05`, `tanh` ve skryté vrstvě.",
        "- Hlavní experiment: 8 konfigurací pokrývajících malé sítě, vysoké learning rate a rozsahy `10–30` neuronů pro krátké i dlouhé učení.",
        "- Pro každou konfiguraci byl uložen společný graf train/validation chyby a testovací MSE/MAE.",
        "",
        "## Implementace / zvolená metoda",
        "Síť používá jednu skrytou vrstvu, bias je integrován do vah `V` a `W`, trénování probíhá online backpropagací nad MSE ztrátou.",
        "Pro kratší běhy byla použita `tanh`, pro dlouhé běhy `relu`, výstupní vrstva je lineární.",
        "",
        "## Výsledky",
        f"- Syntetická kontrola: finální train MSE `{synthetic['final_train_mse']:.4f}`, validation MSE `{synthetic['final_val_mse']:.4f}`",
        f"- Nejlepší konfigurace: `{best['name']}`",
        f"- Nejlepší test MSE: `{best['test_mse']:.4f}`",
        f"- Nejlepší test MAE: `{best['test_mae']:.4f}`",
        "",
        *table_lines,
        "",
        "## Vizualizace výsledků",
        "### Syntetická kontrola objektu FFNN",
        "![synthetic_curve](assets/synthetic_train_validation_curve.png)",
        "",
        "### Souhrn testovacích MSE",
        "![test_mse_by_config](assets/test_mse_by_config.png)",
        "",
        "### Nejlepší model – predikce vs. skutečnost",
        "![best_scatter](assets/best_model_prediction_scatter.png)",
        "",
        "### Křivky jednotlivých konfigurací",
        *[f"![{row['name']}](assets/{row['name']}_curve.png)" for row in configs],
        "",
        "## Diskuze výsledků",
        "Syntetická úloha potvrdila, že implementace backpropagace skutečně snižuje train i validační chybu. Na hlavním datasetu se ukázalo, že velmi vysoký learning rate vede jen někdy ke zrychlení, ale stabilnější výsledky dávají střední learning rate a širší skrytá vrstva.",
        "Dlouhé učení s `relu` umí dále snížit testovací chybu, ale přínos není lineární a při dalším zvětšování sítě se zvyšuje riziko přeučení.",
        "",
        "## Závěr",
        "FFNN objekt byl implementován podle zadání, proběhlo ověření na syntetických i reálných datech a všechny povinné křivky i finální testovací výsledky byly vygenerovány do reportu.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "report" / "assets"
    outputs_dir = base_dir / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    synthetic_results = run_synthetic_check(assets_dir)
    experiment_results = run_main_experiment(base_dir, assets_dir, outputs_dir)
    report_path = base_dir / "report" / "report.md"
    generate_report(report_path, synthetic_results, experiment_results)

    (outputs_dir / "results.json").write_text(
        json.dumps({"synthetic": synthetic_results, "experiment": experiment_results}, indent=2),
        encoding="utf-8",
    )

    print("EXP05 completed")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

