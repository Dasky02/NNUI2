# EXP08 – CNN a velikost jádra

Experiment porovnává jednu PyTorch CNN architekturu nad datasetem `sklearn.datasets.load_digits`.
Mezi variantami se mění pouze `kernel_size` první konvoluce a odpovídající padding.

Spuštění:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
```

Hlavní výstupy:

- `EXP08/results/metrics.csv`
- `EXP08/results/kernel_comparison.csv`
- `EXP08/results/kernel_boxplot_or_barplot.png`
- `EXP08/results/best_confusion_matrix.png`
- `EXP08/results/feature_maps.png`
- `EXP08/results/best_model.pt`
