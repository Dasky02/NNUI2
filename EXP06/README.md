# EXP06 – FFNN klasifikace

Experiment pro cvičení 6 obsahuje:

- pět různých topologií FFNN pro klasifikaci
- deset běhů pro každou topologii s různou inicializací
- boxplot testovací chyby nad topologiemi
- skutečnou PyTorch FFNN implementaci přes `torch.nn.Module`
- confusion matrix, loss curve a uložený nejlepší model

## Spuštění

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP06
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
```

## Výstupy

- `EXP06/results/metrics.csv`
- `EXP06/results/boxplot_test_error.png`
- `EXP06/results/confusion_matrix_best.png`
- `EXP06/results/best_model.pt`
- `EXP06/results/summary.json`
- `EXP06/results/test_predictions_best.csv`
- `EXP06/report/report.md`
