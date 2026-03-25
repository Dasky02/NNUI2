# EXP02 – Jednoduchý perceptron

Implementace splňuje zadání cvičení 2:

- objekt perceptronu s metodami `activation`, `predict`, `train`, `test`, `save`, `load`
- 10 trénování se stejnými hyperparametry a různou inicializací vah
- veřejný dataset `Breast Cancer Wisconsin`
- uložení průběhů chyb, vah, boxplotu, grafu průběhu učení a matice záměn
- vygenerovaný markdown deník v [report/report.md](/Users/dasky/PycharmProjects/NNUI2/EXP02/experiment_02/report/report.md)

## Spuštění

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP02/experiment_02
python3 main.py
```

## Výstupy

- `outputs/weights_run_*.npy`
- `outputs/errors_run_*.npy`
- `outputs/best_model_weights.npy`
- `outputs/run_summary.csv`
- `outputs/metrics.json`
- `report/assets/training_error_runs.png`
- `report/assets/final_error_boxplot.png`
- `report/assets/confusion_matrix.png`
- `report/report.md`
