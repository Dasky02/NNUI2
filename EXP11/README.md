# EXP11 - segmentace Potatoes_seg

Experiment pouziva lokalni segmentacni dataset:

`/Users/dasky/PycharmProjects/cviceni/Potatoes_seg/`

Dataset obsahuje `data.yaml`, pripravene `train/valid/test` rozdeleni, obrazky a
YOLO polygonove segmentacni labely pro jednu tridu `potatoes-`. Soubor
`/Users/dasky/PycharmProjects/cviceni/segmentace.txt` je zohlednen jako puvodni
YOLO26-seg instrukcni material.

Aktualni beh je kratky smoke test (`PASS_WITH_LIMITATIONS`): polygonove anotace se
prevadeji na binarni masky a nad nimi se trenuje mala PyTorch encoder-decoder CNN.
Nejde o dlouhy plny YOLO26-seg trenink.

Spusteni:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
```

Audit bez trenovani:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py --audit-only
```

Hlavni vystupy:

- `EXP11/results/metrics.csv`
- `EXP11/results/results.json`
- `EXP11/results/segmentation_samples.png`
- `EXP11/results/best_model.pt`
