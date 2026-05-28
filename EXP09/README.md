# EXP09 – CNN pro klasifikační úlohy

Experiment porovnava pet skutecne odlisnych PyTorch CNN architektur nad lokalnim
flower classification datasetem:

`/Users/dasky/PycharmProjects/cviceni/dataset/`

Tridy jsou `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`. Varianty meni pocet
konvolucnich vrstev, pooling, dropout, aktivacni funkci a velikost plne propojene
casti pri stejnem train/validation/test splitu a stejnych trenovacich parametrech.

Aktualni konfigurace je kratky kontrolni beh (`PASS_WITH_LIMITATIONS`): 5 architektur,
resize obrazku na 64x64, normalizace podle train splitu a maximalne 8 epoch.

Spuštění:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
```

Hlavní výstupy:

- `EXP09/results/metrics.csv`
- `EXP09/results/architecture_comparison.png`
- `EXP09/results/best_confusion_matrix.png`
- `EXP09/results/loss_curves.png`
- `EXP09/results/prediction_examples.png`
- `EXP09/results/best_model.pt`
