# NNUI2 - Cviceni 8: CNN a velikost konvolucniho jadra

## 1. Cil experimentu
Cilem je porovnat vliv velikosti konvolucniho kernelu v jedne CNN architekture na klasifikaci obrazku cislic. Mezi variantami se meni pouze `kernel_size` a odpovidajici `padding`, aby rozmery feature map zustaly korektni.

## Zdrojove materialy
- Zadani/material cviceni: `/Users/dasky/PycharmProjects/cviceni/NNUI2_08_Cv.pdf`.
- Dostupna zdrojova slozka: `/Users/dasky/PycharmProjects/cviceni/CNN I-20260522/`.
- Poznamka k pouziti: slozka `CNN I-20260522/` slouzi jako referencni material k CNN I a konvolucnim kernelum. Aktualni EXP08 tyto skripty primo neimportuje ani nespousti; vlastni PyTorch implementace ale resi stejne zadani: jedna CNN architektura, pet velikosti kernelu a stejne trenovaci podminky mezi variantami.

## 2. Dataset
- Nazev datasetu: `sklearn.datasets.load_digits`.
- Pocet trid: `10`.
- Pocet obrazku: `1797`.
- Velikost obrazku: `1x8x8 grayscale`.
- Train/val/test split: `1077/360/360` = `60/20/20 stratified`.
- Preprocessing: `pixel values scaled from 0-16 to 0-1; channel dimension added`.

## 3. Architektura CNN
Architektura je stejna pro vsechny varianty: `Conv2d(1, 8, kernel_size=k, padding=k//2) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(feature_dim, 32) -> ReLU -> Linear(32, 10)`. Pocet filtru, pooling, fully-connected cast, loss i optimizer zustavaji stejne.

## 4. Testovane varianty kernelu
| Varianta | Kernel size | Padding | Pocet filtru |
| --- | ---: | ---: | ---: |
| `kernel_1` | 1x1 | 0 | 8 |
| `kernel_3` | 3x3 | 1 | 8 |
| `kernel_5` | 5x5 | 2 | 8 |
| `kernel_7` | 7x7 | 3 | 8 |
| `kernel_9` | 9x9 | 4 | 8 |

## 5. Trenovaci parametry
- Optimizer: `Adam`.
- Learning rate: `0.003`.
- Batch size: `32`.
- Epochs: maximalne `80`, early stopping patience `12` podle validation accuracy.
- Loss function: `CrossEntropyLoss`.
- Seed: zakladni seed `8080`, jednotlive varianty pouzivaji `SEED + kernel_size`.
- Device: `cpu`.

## 6. Vysledky
| Kernel | Accuracy | Precision | Recall | F1-score | Test loss | Epochs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x1 | 0.8556 | 0.8570 | 0.8550 | 0.8519 | 0.3984 | 47 |
| 3x3 | 0.9778 | 0.9782 | 0.9775 | 0.9775 | 0.0792 | 37 |
| 5x5 | 0.9639 | 0.9650 | 0.9636 | 0.9637 | 0.1185 | 41 |
| 7x7 | 0.9639 | 0.9658 | 0.9636 | 0.9638 | 0.1118 | 26 |
| 9x9 | 0.9750 | 0.9756 | 0.9747 | 0.9746 | 0.0881 | 65 |

## 7. Nejlepsi model
Nejlepsi model byl vybran podle nejvyssi test accuracy; pri shode rozhoduje vyssi F1-score a nizsi test loss.
- Nejlepsi kernel: `3x3`.
- Metriky: accuracy `0.9778`, precision `0.9782`, recall `0.9775`, F1 `0.9775`, test loss `0.0792`.
- Ulozeny model: `EXP08/results/best_model.pt`.

## 8. Confusion matrix
Confusion matrix nejlepsiho modelu je ulozena v `EXP08/results/best_confusion_matrix.png`.

![Confusion matrix](EXP08/results/best_confusion_matrix.png)

## 9. Feature mapy
Feature mapy prvniho filtru pro vsech pet kernelu jsou ulozeny v `EXP08/results/feature_maps.png`.

![Feature maps](EXP08/results/feature_maps.png)

## 10. Diskuze
Kernel `1x1` nevidi prostorove okoli pixelu, proto funguje hlavne jako kanalova transformace nad jednim vstupnim kanalem a ma omezenou schopnost zachytit tahy cislic. Kernely `3x3` a `5x5` lepe zachycuji lokalni tvary, hrany a kratke useky tahu.
Vetsi kernely `7x7` a `9x9` maji sirsi receptive field uz v prvni vrstve, ale na obrazech 8x8 mohou prilis rychle agregovat velkou cast obrazku. To muze pomoci potlacit sum, ale soucasne hrozi ztrata jemnych detailu mezi podobnymi cislicemi.
Protoze je padding nastaven na `kernel_size // 2`, rozmery po konvoluci zustavaji srovnatelne a rozdily ve vysledcich lze interpretovat hlavne jako vliv velikosti kernelu. Ostatni podminky, vcetne splitu, optimizeru, learning rate, batch size a architektury klasifikatoru, zustaly stejne.

## 11. Zaver
EXP08 splnuje zadani: pouziva mensi verejny obrazovy dataset, jednu PyTorch CNN architekturu, pet variant kernelu `1x1`, `3x3`, `5x5`, `7x7`, `9x9`, stejne trenovaci podminky, CSV metriky, porovnavaci graf, confusion matrix, feature mapy a ulozeny nejlepsi model.
