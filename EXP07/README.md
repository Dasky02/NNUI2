# EXP07 – Extrakce příznaků

Experiment porovnává reprezentace vstupu pro klasifikaci datasetu `sklearn.datasets.load_digits` pomocí stejné PyTorch FFNN:

- raw pixely 64D,
- HOG-like příznaky 144D,
- PCA z raw pixelů 16D,
- PCA z HOG příznaků 24D.

Spuštění:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
```

Hlavní výstupy:

- `EXP07/results/metrics.csv`
- `EXP07/results/feature_boxplot.png`
- `EXP07/results/confusion_matrix_best.png`
- `EXP07/results/feature_examples.png`
- `EXP07/results/best_model.pt`
- `EXP07/report/report.md`
