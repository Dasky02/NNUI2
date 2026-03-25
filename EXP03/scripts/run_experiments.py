"""CLI entrypoint for NNUI2 Hopfield experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from hopfield.experiments import generate_report, run_mnist, run_toy_3x3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hopfield experiments and generate report.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/assets"),
        help="Directory for generated PNG assets (default: report/assets)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic runs")
    parser.add_argument(
        "--mnist-npz",
        type=Path,
        default=None,
        help="Optional path to local keras-style mnist.npz (x_train/y_train) for offline runs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mnist_npz is not None:
        os.environ["MNIST_NPZ_PATH"] = str(args.mnist_npz.expanduser().resolve())

    toy_seed = int(args.seed) + 81
    toy_results = run_toy_3x3(out_dir, seed=toy_seed)

    mnist_results: dict[str, Any] | None = None
    try:
        mnist_results = run_mnist(out_dir, seed=int(args.seed))
    except RuntimeError as exc:
        print(f"[WARN] MNIST experiment skipped: {exc}")

    report_path = out_dir.parent / "report.md"
    generate_report(report_path, toy_results=toy_results, mnist_results=mnist_results)
    print(f"Assets saved to: {out_dir}")
    print(f"Report generated: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
