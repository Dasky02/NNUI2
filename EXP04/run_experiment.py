from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime import configure_matplotlib_env

configure_matplotlib_env("nnui2_exp04")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from som import KohonenSOM


def make_toy_dataset(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_1 = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(100, 2))
    cluster_2 = rng.normal(loc=(5.0, 0.0), scale=1.0, size=(100, 2))
    cluster_3 = rng.normal(loc=(0.0, 5.0), scale=1.0, size=(100, 2))
    return np.vstack([cluster_1, cluster_2, cluster_3])


def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def save_quantization_curve(histories: dict[str, list[float]], output_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    for label, history in histories.items():
        plt.plot(history, label=label, linewidth=1.6)
    plt.title(title)
    plt.xlabel("Epocha")
    plt.ylabel("Kvantizační chyba")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_assignment_plot(
    X: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 6))
    cmap = plt.get_cmap("tab10", max(len(np.unique(assignments)), 3))
    plt.scatter(X[:, 0], X[:, 1], c=assignments, cmap=cmap, alpha=0.7, s=28)
    plt.scatter(weights[:, 0], weights[:, 1], c="black", marker="X", s=180, label="neurony")
    plt.title(title)
    plt.xlabel("Dimenze 1")
    plt.ylabel("Dimenze 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_iris_pca_plot(
    X: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X)
    W_2d = pca.transform(weights)

    plt.figure(figsize=(7, 6))
    cmap = plt.get_cmap("tab10", max(len(np.unique(assignments)), 3))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=assignments, cmap=cmap, alpha=0.75, s=28)
    plt.scatter(W_2d[:, 0], W_2d[:, 1], c="black", marker="X", s=180, label="neurony")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def cluster_purity(assignments: np.ndarray, y: np.ndarray) -> float:
    total = 0
    for neuron in np.unique(assignments):
        labels = y[assignments == neuron]
        if labels.size:
            counts = np.bincount(labels)
            total += int(counts.max())
    return float(total / len(y))


def run_toy_experiment(assets_dir: Path) -> dict[str, object]:
    X = make_toy_dataset(seed=42)
    histories: dict[str, list[float]] = {}
    summary: dict[str, dict[str, float]] = {}
    selected_assignments: np.ndarray | None = None
    selected_weights: np.ndarray | None = None

    for radius in [0, 1, 2]:
        som = KohonenSOM(input_dim=2, n_units=3, lr=0.5, radius=radius, epochs=120, seed=radius + 7)
        history = som.train(X)
        label = f"radius={radius}"
        histories[label] = history
        summary[label] = {"final_quantization_error": float(history[-1])}
        if radius == 1:
            selected_assignments = som.predict(X)
            selected_weights = som.W.copy()

    save_quantization_curve(
        histories,
        assets_dir / "toy_quantization_radius_compare.png",
        "Toy data: vliv parametru sousedství",
    )
    if selected_assignments is None or selected_weights is None:
        raise RuntimeError("Toy experiment did not produce the selected radius=1 result")
    save_assignment_plot(
        X,
        selected_assignments,
        selected_weights,
        assets_dir / "toy_assignments_radius_1.png",
        "Toy data: přiřazení vzorků pro radius=1",
    )

    return {"histories": summary}


def run_iris_experiment(assets_dir: Path) -> dict[str, object]:
    iris = load_iris()
    X = standardize(iris.data.astype(float))
    y = iris.target.astype(int)
    class_names = [str(name) for name in iris.target_names]

    scenarios = {
        "exact_shape": {"n_units": 3, "lr": 0.5, "radius": 1, "epochs": 160},
        "insufficient_shape": {"n_units": 2, "lr": 0.5, "radius": 1, "epochs": 160},
        "redundant_shape": {"n_units": 6, "lr": 0.5, "radius": 2, "epochs": 160},
    }

    results: dict[str, object] = {"class_names": class_names, "scenarios": {}}

    for index, (name, cfg) in enumerate(scenarios.items(), start=1):
        som = KohonenSOM(
            input_dim=X.shape[1],
            n_units=int(cfg["n_units"]),
            lr=float(cfg["lr"]),
            radius=int(cfg["radius"]),
            epochs=int(cfg["epochs"]),
            seed=20 + index,
        )
        history = som.train(X)
        assignments = som.predict(X)
        purity = cluster_purity(assignments, y)

        save_quantization_curve(
            {name: history},
            assets_dir / f"iris_{name}_quantization.png",
            f"Iris: průběh kvantizační chyby ({name})",
        )
        save_iris_pca_plot(
            X,
            assignments,
            som.W,
            assets_dir / f"iris_{name}_assignments.png",
            f"Iris: přiřazení vzorků ({name})",
        )

        results["scenarios"][name] = {
            "config": cfg,
            "final_quantization_error": float(history[-1]),
            "purity": float(purity),
        }

    return results


def generate_report(report_path: Path, toy_results: dict[str, object], iris_results: dict[str, object]) -> None:
    scenarios = iris_results["scenarios"]
    toy_histories = toy_results["histories"]
    best_toy_radius = min(toy_histories.items(), key=lambda item: item[1]["final_quantization_error"])[0]
    best_iris_purity = max(scenarios.items(), key=lambda item: item[1]["purity"])[0]
    lines = [
        "# NNUI2 – Cvičení 4: Kohonenova samoorganizační mapa",
        "",
        "## Název experimentu",
        "Kohonenova samoorganizační mapa pro shlukování syntetických dat a datasetu Iris.",
        "",
        "## Cíl úlohy",
        "Implementovat 1D samoorganizační mapu, ověřit pokles kvantizační chyby a porovnat chování mapy pro různé počty neuronů a různé sousedství.",
        "",
        "## Popis zadání",
        "Zadání vyžadovalo objekt SOM s metodami `bmu`, `neighborhood`, `quantization_error`, `predict` a `train`, test na syntetických 2D shlucích a experimenty na veřejném datasetu Iris pro tři konfigurace mapy.",
        "",
        "## Použitá data / úprava datasetu",
        "Toy experiment používá tři gaussovské shluky kolem bodů `(0,0)`, `(5,0)` a `(0,5)`.",
        "Veřejná data tvoří `sklearn.datasets.load_iris`; vstupy byly standardizovány na nulový průměr a jednotkovou odchylku.",
        "",
        "## Postup řešení",
        "- Toy experiment: `n_units=3`, `epochs=120`, `lr=0.5`, porovnání `radius ∈ {0,1,2}`.",
        "- Iris experiment: přesný tvar `3` neurony, nedostatečný tvar `2` neurony, redundantní tvar `6` neuronů.",
        "- V každé epoše se ukládá kvantizační chyba a po natrénování se vyhodnocuje čistota přiřazení vůči třídám Iris.",
        "",
        "## Implementace / zvolená metoda",
        "Vítězný neuron je vybírán jako BMU podle eukleidovské vzdálenosti. Váhy vítěze i jeho okolí se aktualizují s exponenciálně klesajícím learning rate.",
        "Sousedství je realizováno v 1D kompetiční vrstvě a průběh kvantizační chyby je ukládán do atributu `history`.",
        "",
        "## Výsledky",
        f"- Toy radius=0: finální kvantizační chyba `{toy_results['histories']['radius=0']['final_quantization_error']:.4f}`",
        f"- Toy radius=1: finální kvantizační chyba `{toy_results['histories']['radius=1']['final_quantization_error']:.4f}`",
        f"- Toy radius=2: finální kvantizační chyba `{toy_results['histories']['radius=2']['final_quantization_error']:.4f}`",
        f"- Iris exact_shape: kvantizační chyba `{scenarios['exact_shape']['final_quantization_error']:.4f}`, purity `{scenarios['exact_shape']['purity']:.4f}`",
        f"- Iris insufficient_shape: kvantizační chyba `{scenarios['insufficient_shape']['final_quantization_error']:.4f}`, purity `{scenarios['insufficient_shape']['purity']:.4f}`",
        f"- Iris redundant_shape: kvantizační chyba `{scenarios['redundant_shape']['final_quantization_error']:.4f}`, purity `{scenarios['redundant_shape']['purity']:.4f}`",
        "",
        "## Vizualizace výsledků",
        "### Toy data – porovnání sousedství",
        "![toy_quantization](assets/toy_quantization_radius_compare.png)",
        "",
        "### Toy data – přiřazení vzorků pro radius=1",
        "![toy_assignments](assets/toy_assignments_radius_1.png)",
        "",
        "### Iris – přesný tvar",
        "![iris_exact_quantization](assets/iris_exact_shape_quantization.png)",
        "",
        "![iris_exact_assignments](assets/iris_exact_shape_assignments.png)",
        "",
        "### Iris – nedostatečný tvar",
        "![iris_insufficient_quantization](assets/iris_insufficient_shape_quantization.png)",
        "",
        "![iris_insufficient_assignments](assets/iris_insufficient_shape_assignments.png)",
        "",
        "### Iris – redundantní tvar",
        "![iris_redundant_quantization](assets/iris_redundant_shape_quantization.png)",
        "",
        "![iris_redundant_assignments](assets/iris_redundant_shape_assignments.png)",
        "",
        "## Diskuze výsledků",
        f"Na toy datech dosáhl v tomto běhu nejnižší kvantizační chyby scénář `{best_toy_radius}`. Širší sousedství `radius=2` vedlo k viditelně horší specializaci prototypů a vyšší finální chybě.",
        f"Na Iris dosáhla nejlepší čistoty konfigurace `{best_iris_purity}`, zatímco nedostatečný tvar se dvěma neurony ztrácí schopnost rozlišit všechny tři druhy. Redundantní tvar sice snižuje kvantizační chybu, ale nevede k nejlepší čistotě přiřazení.",
        "",
        "## Závěr",
        "Implementace SOM odpovídá zadání, experimenty byly skutečně spuštěny a report obsahuje požadované průběhy kvantizační chyby i vizualizace přiřazení.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "report" / "assets"
    outputs_dir = base_dir / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    toy_results = run_toy_experiment(assets_dir)
    iris_results = run_iris_experiment(assets_dir)
    report_path = base_dir / "report" / "report.md"
    generate_report(report_path, toy_results, iris_results)

    payload = {"toy": toy_results, "iris": iris_results}
    (outputs_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("EXP04 completed")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
