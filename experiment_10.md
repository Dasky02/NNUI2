# NNUI2 - Cviceni 10: YOLO OBB detekce

## 1. Cil experimentu
Cilem je porovnat pet variant YOLO OBB detekce nad OBB casti Bricks Detection Datasetu a vybrat nejlepsi model podle `mAP50-95`.

## 2. Dataset
- Dataset: `Bricks Detection Dataset` ze Zenodo.
- DOI / zdroj: `10.5281/zenodo.18952529`, `https://zenodo.org/records/18952529`.
- Pouzita cast: `OBB`.
- Segmentation cast neni pouzita, protoze EXP10 resi detekci orientovanych bounding boxu, ne segmentacni masky.
- YAML: `/Users/dasky/PycharmProjects/cviceni/datasets/Bricks_Detection_Dataset/OBB/data_all.yaml`.
- Stav auditu datasetu: `ok`. 
- Pocet trid: `3`; nazvy: `{0: 'InvalidBrick', 1: 'Pallete', 2: 'Brick'}`.
- Pocet obrazku train/val/test: `83/20/30`.
- Pocet labelu train/val/test: `83/20/30`.
- Potvrzeni OBB formatu 9 hodnot: `True`.

## 3. Varianty modelu
| Varianta | Pozadovany model | Realny model | Image size | Epochs | Batch | LR | Augmentace | Hardware |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `V1_yolo26n_obb_img80` | `yolo26n-obb.pt` | `yolo11n-obb.pt` | 80 | 1 | 8 | 0.01 | default | cpu |
| `V2_yolo26n_obb_img240` | `yolo26n-obb.pt` | `yolo11n-obb.pt` | 240 | 1 | 8 | 0.01 | default | cpu |
| `V3_yolo26s_obb_img240` | `yolo26s-obb.pt` | `yolo11s-obb.pt` | 240 | 1 | 8 | 0.01 | default | cpu |
| `V4_yolo26s_obb_img320` | `yolo26s-obb.pt` | `yolo11s-obb.pt` | 320 | 1 | 8 | 0.01 | default | cpu |
| `V5_yolo26n_obb_img320_lr005_aug` | `yolo26n-obb.pt` | `yolo11n-obb.pt` | 320 | 1 | 8 | 0.005 | mosaic=0.5, degrees=15 | cpu |

## 4. Trenovani
- Rezim behu: `smoke`.
- Skript podporuje `--data PATH`, `--audit-only`, `--smoke` a `--epochs N`.
- Pokud presny YOLO26 model neni dostupny v ultralytics, skript zkusi kompatibilni OBB model stejne velikosti (`yolo11*-obb.pt`, potom `yolov8*-obb.pt`) a realny model zapise do CSV/reportu.
- Vystupy YOLO se ukladaji do `EXP10/results/yolo_runs/`.

## 5. Vysledky
Tabulka uvadi validacni metriky z trenovani a samostatnou evaluaci nad test splitem. Nejlepsi model se vybira podle test `mAP50-95`, pokud je k dispozici.
| Varianta | Val P | Val R | Val mAP50 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test mAP50-95 | Train loss | Validation loss | Stav |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `V1_yolo26n_obb_img80` | 0.01488 | 0.10181 | 0.01062 | 0.0024 | 0.011974 | 0.022094 | 0.0068087 | 0.0013544 | 5.2575 | 4.5464 | completed |
| `V2_yolo26n_obb_img240` | 0.08957 | 0.31694 | 0.07493 | 0.02377 | 0.088816 | 0.33184 | 0.068636 | 0.020922 | 3.7968 | 2.6852 | completed |
| `V3_yolo26s_obb_img240` | 0.06901 | 0.34846 | 0.05973 | 0.02419 | 0.077958 | 0.31255 | 0.060503 | 0.025213 | 3.6744 | 2.0162 | completed |
| `V4_yolo26s_obb_img320` | 0.11131 | 0.38652 | 0.1314 | 0.05471 | 0.1194 | 0.35565 | 0.11049 | 0.050603 | 3.4081 | 1.8059 | completed |
| `V5_yolo26n_obb_img320_lr005_aug` | 0.1193 | 0.42109 | 0.09294 | 0.05124 | 0.12241 | 0.43109 | 0.097978 | 0.054124 | 2.8223 | 1.7102 | completed |

## 6. Nejlepsi model
- Nejlepsi varianta: `V5_yolo26n_obb_img320_lr005_aug`.
- Kriterium: nejvyssi test `mAP50-95 = 0.054124`.
- Best weights: `/Users/dasky/PycharmProjects/NNUI2/EXP10/results/yolo_runs/V5_yolo26n_obb_img320_lr005_aug/weights/best.pt`.

## 7. Confusion matrix a ukazka detekce
- Pokud YOLO test evaluace vytvori confusion matrix, nejlepsi se kopiruje do `EXP10/results/best_confusion_matrix.png`.
- Pokud YOLO test evaluace vytvori predikcni obrazek, nejlepsi se kopiruje do `EXP10/results/test_prediction_best.png`.

## 8. Diskuze
Mensi image size `80` slouzi hlavne jako rychly baseline/smoke. Varianty `240` a `320` by mely lepe zachytit polohu rohu OBB, ale jsou pomalejsi. Vetsi model `s` muze mit lepsi kapacitu nez `n`, ale muze byt narocnejsi na CPU/GPU. Varianta V5 meni learning rate a augmentaci, aby bylo videt, zda pomuze robustnosti.

## 9. Zaver
Report neobsahuje vymyslene metriky. Pokud je dataset nebo trenink nedostupny, radky zustavaji `not_run` nebo `failed` s prazdnymi metrikami. Pokud probehl jen smoke beh, stav experimentu je `PASS_WITH_LIMITATIONS`.
