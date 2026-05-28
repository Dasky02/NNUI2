# EXP10 - YOLO26 OBB detekce

Skript pouziva lokalni OBB cast Bricks Detection Datasetu ze Zenodo
`10.5281/zenodo.18952529`. Segmentation cast datasetu se pro EXP10 nepouziva,
protoze toto cviceni resi oriented bounding box detekci.

Pouzity dataset:

```text
/Users/dasky/PycharmProjects/cviceni/datasets/Bricks_Detection_Dataset/OBB/data_all.yaml
```

Audit overuje strukturu `train/val/test`, existenci `images` a `labels` a
format OBB labelu `class x1 y1 x2 y2 x3 y3 x4 y4` s 9 hodnotami na radek. Bez
validniho OBB datasetu skript negeneruje falesne metriky.

Spuštění auditu bez trénování:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
```

Smoke beh nad vsemi 5 variantami:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py --data /Users/dasky/PycharmProjects/cviceni/datasets/Bricks_Detection_Dataset/OBB/data_all.yaml --smoke --epochs 1 --device cpu
```

Pozadovane modely jsou `yolo26n-obb.pt` a `yolo26s-obb.pt`. Pokud nejsou v
aktualni instalaci Ultralytics dostupne, skript pouzije kompatibilni OBB
fallback (`yolo11n-obb.pt` nebo `yolo11s-obb.pt`) a zapise to do CSV/reportu.

Hlavní výstupy:

- `EXP10/results/metrics.csv`
- `EXP10/results/variant_summary.csv`
- `EXP10/results/dataset_check.json`
- `EXP10/results/results.json`
- `EXP10/results/best_model.pt` po úspěšném trénování
- `EXP10/results/best_confusion_matrix.png` po úspěšné validaci
- `EXP10/results/test_prediction_best.png` po úspěšné test predikci
