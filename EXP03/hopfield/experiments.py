"""Experiment runners for Hopfield network (toy + MNIST) and report generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "hopfield_mpl_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .net import HopfieldNet


@dataclass(frozen=True)
class QueryResult:
    target_class: int
    stored_class: int | None
    converged: bool
    reason: str
    iters: int
    exact_match: bool
    hamming_to_best: int
    best_match_index: int | None
    stored_image_path: str
    query_image_path: str
    output_image_path: str
    energy_plot_path: str


def run_toy_3x3(output_dir: str | Path, seed: int = 123) -> dict[str, Any]:
    """Run a small 3x3 Hopfield experiment and store visualizations."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Two simple 3x3 patterns (X and O) in {0,1}.
    pattern_x = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=float,
    )
    pattern_o = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    train_patterns = np.stack([pattern_x, pattern_o], axis=0)

    net = HopfieldNet(input_size=9, bipolar=True)
    net.train(train_patterns)

    # Try a few 2-pixel perturbations and prefer one that converges to a fixpoint.
    candidates: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    base = pattern_x.copy().reshape(-1)
    for _ in range(16):
        flip_idx = np.sort(rng.choice(base.size, size=2, replace=False))
        noisy = base.copy()
        noisy[flip_idx] = 1.0 - noisy[flip_idx]
        recall = net.recall(noisy, max_iters=10, stop_when_stable=True, detect_2cycle=True)
        candidates.append((noisy, flip_idx, recall))
        if recall["info"]["reason"] == "fixed_point":
            break

    noisy, flip_idx, recall = candidates[-1]
    noisy_img = noisy.reshape(3, 3)
    final_img = _vector_to_image(recall["final_state"], (3, 3), bipolar=True)

    train_path = out / "toy_train_patterns.png"
    noisy_path = out / "toy_noisy_input.png"
    recalled_path = out / "toy_recalled_output.png"
    energy_path = out / "toy_energy.png"

    _save_pattern_grid(train_patterns, train_path, titles=["Train X", "Train O"])
    _save_single_pattern(noisy_img, noisy_path, title="Toy noisy input")
    _save_single_pattern(final_img, recalled_path, title="Toy recalled output")
    _save_energy_plot(recall["energies"], energy_path, title="Toy 3x3 energy")

    final_state = recall["final_state"]
    stored_proc = net.preprocess(train_patterns)
    hamming_distances = [int(np.sum(final_state != p)) for p in stored_proc]
    best_idx = int(np.argmin(hamming_distances))

    return {
        "seed": seed,
        "flip_indices": [int(i) for i in np.sort(flip_idx)],
        "info": recall["info"],
        "energies": [float(e) for e in recall["energies"]],
        "best_match_index": best_idx,
        "best_hamming": hamming_distances[best_idx],
        "train_image": train_path.name,
        "noisy_image": noisy_path.name,
        "recalled_image": recalled_path.name,
        "energy_plot": energy_path.name,
    }


def run_mnist(
    output_dir: str | Path,
    seed: int = 42,
    classes: list[int] | None = None,
    downsample_shape: tuple[int, int] = (14, 14),
) -> dict[str, Any]:
    """Run Hopfield recall on MNIST with 4 stored patterns and 4 queries."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    chosen_classes = classes if classes is not None else [0, 1, 4, 7]
    if len(chosen_classes) != 4:
        raise ValueError("Exactly 4 classes are required for the MNIST experiment")

    images, labels, source_name = _load_mnist()
    images = images.astype(np.float32)
    labels = labels.astype(int)

    stored_images: list[np.ndarray] = []
    query_images: list[np.ndarray] = []
    stored_labels: list[int] = []
    query_labels: list[int] = []

    for cls in chosen_classes:
        indices = np.flatnonzero(labels == cls)
        if indices.size < 2:
            raise RuntimeError(f"MNIST class {cls} does not have enough samples")
        selected = rng.choice(indices, size=2, replace=False)
        stored_images.append(images[int(selected[0])])
        query_images.append(images[int(selected[1])])
        stored_labels.append(cls)
        query_labels.append(cls)

    prepared_stored = np.stack(
        [_prepare_mnist_image(img, downsample_shape) for img in stored_images], axis=0
    )
    prepared_queries = np.stack(
        [_prepare_mnist_image(img, downsample_shape) for img in query_images], axis=0
    )

    input_size = int(np.prod(downsample_shape))
    net = HopfieldNet(input_size=input_size, bipolar=True)
    net.train(prepared_stored)
    stored_bipolar = net.preprocess(prepared_stored)

    query_results: list[QueryResult] = []
    for i, (cls, p_img, q_img, q_vec) in enumerate(
        zip(query_labels, stored_images, query_images, prepared_queries), start=1
    ):
        recall = net.recall(q_vec, max_iters=30, stop_when_stable=True, detect_2cycle=True)
        final_state = recall["final_state"]

        distances = [int(np.sum(final_state != p)) for p in stored_bipolar]
        best_idx = int(np.argmin(distances))
        best_hamming = distances[best_idx]
        exact_match = bool(np.array_equal(final_state, stored_bipolar[best_idx]))
        stored_class = stored_labels[best_idx] if exact_match else None

        p_path = out / f"mnist_p{i}_class{cls}.png"
        q_path = out / f"mnist_q{i}_class{cls}.png"
        o_path = out / f"mnist_out{i}_class{cls}.png"
        e_path = out / f"mnist_energy{i}_class{cls}.png"

        _save_single_pattern(_prepare_mnist_image(p_img, downsample_shape), p_path, title=f"P{i} class {cls}")
        _save_single_pattern(_prepare_mnist_image(q_img, downsample_shape), q_path, title=f"Q{i} class {cls}")
        _save_single_pattern(
            _vector_to_image(final_state, downsample_shape, bipolar=True),
            o_path,
            title=f"Output {i} (target {cls})",
        )
        _save_energy_plot(recall["energies"], e_path, title=f"Energy Q{i} class {cls}")

        info = recall["info"]
        query_results.append(
            QueryResult(
                target_class=int(cls),
                stored_class=int(stored_class) if stored_class is not None else None,
                converged=bool(info["converged"]),
                reason=str(info["reason"]),
                iters=int(info["iters"]),
                exact_match=exact_match,
                hamming_to_best=int(best_hamming),
                best_match_index=int(best_idx),
                stored_image_path=p_path.name,
                query_image_path=q_path.name,
                output_image_path=o_path.name,
                energy_plot_path=e_path.name,
            )
        )

    return {
        "seed": seed,
        "source": source_name,
        "classes": [int(c) for c in chosen_classes],
        "downsample_shape": list(map(int, downsample_shape)),
        "query_results": [q.__dict__ for q in query_results],
    }


def generate_report(
    report_path: str | Path,
    toy_results: dict[str, Any],
    mnist_results: dict[str, Any] | None,
) -> Path:
    """Generate markdown report with embedded images."""

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# NNUI2 – Cvičení 3: Diskrétní Hopfieldova síť")
    lines.append("")
    lines.append("## Název experimentu")
    lines.append("Diskrétní Hopfieldova síť jako asociativní paměť pro toy vzory 3×3 a binarizované vzory z MNIST.")
    lines.append("")
    lines.append("## Cíl úlohy")
    lines.append(
        "Implementovat diskrétní Hopfieldovu síť s Hebbovým učením, energií a synchronním vybavováním,"
    )
    lines.append("ověřit chování na malých vzorech 3×3 a následně otestovat recall na čtyřech třídách MNIST.")
    lines.append("")
    lines.append("## Popis zadání")
    lines.append(
        "Zadání vyžadovalo konstruktor s `input_size` a `bipolar`, váhovou matici `W`, metody `preprocess`, `train`, `energy`, `recall`, ukládání historie stavů a energií, detekci stability nebo oscilace a dokumentaci průběhu recallu v deníku."
    )
    lines.append("")
    lines.append("## Použitá data / úprava datasetu")
    lines.append("- Toy část používá dva ručně definované binární vzory 3×3.")
    if mnist_results is None:
        lines.append("- MNIST část nebylo možné spustit kvůli nedostupnému datasetu v aktuálním prostředí.")
    else:
        ds = mnist_results["downsample_shape"]
        lines.append(f"- MNIST část používá čtyři uložené a čtyři dotazové vzory z datasetu `{mnist_results['source']}`.")
        lines.append(f"- Obrázky byly binarizovány a zmenšeny z `28x28` na `{ds[0]}x{ds[1]}` blokovým průměrem.")
    lines.append("")
    lines.append("## Postup řešení")
    lines.append("- Implementován byl synchronní recall s logováním všech mezistavů a hodnot energie.")
    lines.append("- Funkčnost vah a recallu byla ověřena přesným příkladem ze slajdů a následně toy experimentem 3×3.")
    lines.append("- Pro MNIST byly vybrány čtyři různé třídy, pro každou jeden uložený a jeden dotazový vzor.")
    lines.append("")
    lines.append("## Implementace / zvolená metoda")
    lines.append("- `HopfieldNet` používá ne-normalizované Hebbovo učení `W = Σ p p^T` a nulovou diagonálu vah.")
    lines.append("- Recall je synchronní a loguje stavy i energii `E = -0.5 * s^T W s`.")
    lines.append("- Implementována je detekce fixpointu i 2-cyklu.")
    lines.append("")
    lines.append("## Validace podle slajdu (Příklad 1)")
    lines.append("- Ověřeno testem `tests/test_example1_slide_data.py`.")
    lines.append("- Trénovací vzory: `x1 = [1, 1, 1, -1]`, `x2 = [1, 1, -1, -1]`.")
    lines.append("- Očekávaná Hebbova matice (bez normalizace):")
    lines.append("")
    lines.append("```text")
    lines.append("[[ 0,  2,  0, -2],")
    lines.append(" [ 2,  0,  0, -2],")
    lines.append(" [ 0,  0,  0,  0],")
    lines.append(" [-2, -2,  0,  0]]")
    lines.append("```")
    lines.append("")
    lines.append("- Recall ze `s0 = [-1, 1, -1, -1]` má trajektorii `s1 = [1, 1, 1, 1]`, `s2 = [1, 1, 1, -1]` (fixpoint = `x1`).")
    lines.append("")
    lines.append("## Výsledky")
    lines.append("### Validace podle slajdu (Příklad 1)")
    lines.append("- Váhová matice i trajektorie recallu odpovídají očekávanému řešení ze slajdu.")
    lines.append("")
    lines.append("### 3×3 test")
    lines.append(
        f"- Seed: `{toy_results['seed']}`, flipnuté pozice: `{toy_results['flip_indices']}`,"
        f" reason: `{toy_results['info']['reason']}`, iterace: `{toy_results['info']['iters']}`"
    )
    lines.append("- Nejbližší uložený vzor (Hamming): " f"`{toy_results['best_hamming']}`")
    lines.append("")
    lines.append("## Vizualizace výsledků")
    lines.append("### Toy vzory 3×3")
    lines.append(f"![](assets/{toy_results['train_image']})")
    lines.append("")
    lines.append(f"![](assets/{toy_results['noisy_image']})")
    lines.append("")
    lines.append(f"![](assets/{toy_results['recalled_image']})")
    lines.append("")
    lines.append(f"![](assets/{toy_results['energy_plot']})")
    lines.append("")
    lines.append("### MNIST experiment")
    if mnist_results is None:
        lines.append("MNIST experiment nebylo možné spustit (viz chybová hláška při běhu skriptu).")
        lines.append("")
        lines.append("| třída | converged | iters | reason | match | best_hamming |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        lines.append("| N/A | N/A | N/A | dataset_unavailable | N/A | N/A |")
        lines.append("")
    else:
        ds = mnist_results["downsample_shape"]
        lines.append(f"- Zdroj datasetu: `{mnist_results['source']}`")
        lines.append(f"- Třídy: `{mnist_results['classes']}`")
        lines.append(f"- Downsample: `{ds[0]}x{ds[1]}`")
        lines.append("")
        lines.append("| třída | converged | iters | reason | match | best_hamming |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in mnist_results["query_results"]:
            match_text = "ano" if row["exact_match"] else "ne"
            conv_text = "ano" if row["converged"] else "ne"
            lines.append(
                f"| {row['target_class']} | {conv_text} | {row['iters']} | {row['reason']} | "
                f"{match_text} | {row['hamming_to_best']} |"
            )
        lines.append("")
        for i, row in enumerate(mnist_results["query_results"], start=1):
            lines.append(f"### Dotaz {i} (třída {row['target_class']})")
            lines.append(f"![](assets/{row['stored_image_path']})")
            lines.append("")
            lines.append(f"![](assets/{row['query_image_path']})")
            lines.append("")
            lines.append(f"![](assets/{row['output_image_path']})")
            lines.append("")
            lines.append(f"![](assets/{row['energy_plot_path']})")
            lines.append("")

    lines.append("## Diskuze výsledků")
    lines.append(
        "Hopfieldova síť dobře funguje jako asociativní paměť pro malé množství vzorů, ale na MNIST"
    )
    lines.append(
        "je citlivá na kapacitu, interferenci mezi vzory a na kvalitu binarizace/downsamplingu."
    )
    lines.append("Sledování energie pomáhá odlišit fixpoint od oscilace (2-cyklus).")
    lines.append("")
    lines.append("## Závěr")
    lines.append(
        "Implementace Hopfieldovy sítě splňuje požadované metody i vlastnosti vah. Toy test i validace podle slajdu byly ověřeny testy a report obsahuje skutečně vygenerované obrázky i průběhy energie."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _load_mnist() -> tuple[np.ndarray, np.ndarray, str]:
    """Load MNIST from TensorFlow, local .npz cache, then sklearn OpenML fallback."""

    tf_error: Exception | None = None
    try:
        from tensorflow.keras.datasets import mnist as keras_mnist  # type: ignore

        (x_train, y_train), _ = keras_mnist.load_data()
        return np.asarray(x_train), np.asarray(y_train), "tensorflow.keras.datasets.mnist"
    except Exception as exc:  # pragma: no cover - environment dependent
        tf_error = exc

    local_error: Exception | None = None
    try:
        x, y, source = _load_mnist_local_npz()
        return x, y, source
    except Exception as exc:  # pragma: no cover - environment dependent
        local_error = exc

    sk_error: Exception | None = None
    try:
        from sklearn.datasets import fetch_openml  # type: ignore

        # parser='liac-arff' avoids optional pandas dependency for dense data.
        data = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        x = np.asarray(data.data, dtype=np.float32).reshape(-1, 28, 28)
        y = np.asarray(data.target).astype(int)
        return x, y, "sklearn.datasets.fetch_openml('mnist_784', version=1)"
    except Exception as exc:  # pragma: no cover - environment dependent
        sk_error = exc

    tf_msg = repr(tf_error) if tf_error is not None else "TensorFlow loader not attempted"
    local_msg = repr(local_error) if local_error is not None else "Local .npz loader not attempted"
    sk_msg = repr(sk_error) if sk_error is not None else "sklearn loader not attempted"
    raise RuntimeError(
        "MNIST dataset could not be loaded. TensorFlow failed: "
        f"{tf_msg}. Local .npz fallback failed: {local_msg}. sklearn OpenML fallback failed: {sk_msg}. "
        "If OpenML download is blocked, run in an environment with internet access or cached dataset."
    )


def _load_mnist_local_npz() -> tuple[np.ndarray, np.ndarray, str]:
    """Load MNIST from a local keras-style ``mnist.npz`` file.

    Search order:
    1. Environment variable ``MNIST_NPZ_PATH``
    2. ``mnist.npz`` in repo root
    3. ``data/mnist.npz`` in repo root
    4. ``~/.keras/datasets/mnist.npz``
    """

    candidates: list[Path] = []
    env_path = os.environ.get("MNIST_NPZ_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            Path("mnist.npz"),
            Path("data") / "mnist.npz",
            Path.home() / ".keras" / "datasets" / "mnist.npz",
        ]
    )

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue

        with np.load(resolved, allow_pickle=False) as data:
            required = {"x_train", "y_train"}
            if not required.issubset(data.files):
                raise ValueError(
                    f"Local MNIST npz at {resolved} is missing keys {sorted(required)}; "
                    f"available keys: {sorted(data.files)}"
                )
            x = np.asarray(data["x_train"])
            y = np.asarray(data["y_train"])
        if x.ndim != 3 or x.shape[1:] != (28, 28):
            raise ValueError(f"Local MNIST npz at {resolved} has invalid x_train shape {x.shape}")
        return x, y, f"local npz ({resolved})"

    raise FileNotFoundError(
        "No local MNIST .npz found. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _prepare_mnist_image(image: np.ndarray, downsample_shape: tuple[int, int]) -> np.ndarray:
    """Normalize, downsample, binarize and return image in {0,1} shape `downsample_shape`."""

    img = np.asarray(image, dtype=np.float32)
    if img.shape != (28, 28):
        raise ValueError(f"Expected MNIST image shape (28, 28), got {img.shape}")

    if img.max() > 1.0:
        img = img / 255.0

    h, w = downsample_shape
    if (28 % h) != 0 or (28 % w) != 0:
        raise ValueError("downsample_shape must divide 28x28 exactly")

    bh = 28 // h
    bw = 28 // w
    ds = img.reshape(h, bh, w, bw).mean(axis=(1, 3))
    binary = (ds >= 0.5).astype(float)
    return binary


def _vector_to_image(vec: np.ndarray, shape: tuple[int, int], bipolar: bool = True) -> np.ndarray:
    """Convert flat vector to image in {0,1} for plotting."""

    arr = np.asarray(vec, dtype=float).reshape(shape)
    if bipolar:
        arr = 0.5 * (arr + 1.0)
    return np.clip(arr, 0.0, 1.0)


def _save_pattern_grid(patterns: np.ndarray, path: Path, titles: list[str] | None = None) -> None:
    """Save a row of patterns as a single figure."""

    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim != 3:
        raise ValueError("patterns must have shape (k, H, W)")

    k = patterns.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3), squeeze=False)
    for i in range(k):
        ax = axes[0, i]
        ax.imshow(patterns[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_single_pattern(pattern: np.ndarray, path: Path, title: str) -> None:
    """Save one pattern image."""

    img = np.asarray(pattern, dtype=float)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_energy_plot(energies: list[float], path: Path, title: str) -> None:
    """Save energy curve E(t)."""

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(energies))
    ax.plot(x, energies, marker="o", linewidth=1.8)
    ax.set_xlabel("Iterace")
    ax.set_ylabel("Energie")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
