# NNUI2 - Cviceni 9: Porovnani CNN architektur

## 1. Cil experimentu
Cilem je porovnat pet ruznych CNN architektur na stejne obrazove klasifikacni uloze a vyhodnotit vliv poctu konvolucnich vrstev, poolingu, dropoutu, aktivacni funkce a velikosti plne propojene casti.

Stav behu: `PASS_WITH_LIMITATIONS`. Experiment byl spusten jako kratky kontrolni beh, aby se overilo pouziti lokalniho flower datasetu a generovani vsech vystupu.

## 2. Dataset
- Nazev datasetu: `Local flower classification dataset`.
- Cesta k datasetu: `/Users/dasky/PycharmProjects/cviceni/dataset`.
- Pocet trid: `5`.
- Tridy: `daisy, dandelion, rose, sunflower, tulip`.
- Pocet obrazku: `2746`.
- Velikost obrazku po resize: `3x64x64 RGB`.
- Preprocessing: `RGB conversion, resize to 64x64, pixel scaling to 0-1, channel-wise normalization using train split mean/std`.
- Train/val/test split: `1647/549/550` = `60/20/20 stratified`.

## 3. Spolecne trenovaci podminky
- Optimizer: `Adam`.
- Learning rate: `0.001`.
- Batch size: `32`.
- Epochs: maximalne `8`, early stopping patience `3` podle validation accuracy.
- Loss function: `CrossEntropyLoss`.
- Seed: zakladni seed `9090`, jednotlive architektury pouzivaji `SEED + index`.
- Device: `cpu`.

## 4. Testovane architektury
| Architektura | Conv vrstvy | Filtry | Kernel size | Pooling | Dropout | Aktivace | FFNN cast |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| `A1_conv1_pool_fc` | 1 | [8] | [3] | po conv indexech [0] | 0.00 | relu | Linear(feature_dim, 48) -> relu -> Linear(48, 5) |
| `A2_conv2_pool_fc` | 2 | [8, 16] | [3, 3] | po conv indexech [1] | 0.00 | relu | Linear(feature_dim, 48) -> relu -> Linear(48, 5) |
| `A3_conv3_pool_fc` | 3 | [8, 16, 16] | [3, 3, 3] | po conv indexech [1] | 0.00 | relu | Linear(feature_dim, 48) -> relu -> Linear(48, 5) |
| `A4_conv2_dropout_fc` | 2 | [8, 16] | [3, 3] | po conv indexech [1] | 0.25 | relu | Linear(feature_dim, 48) -> relu -> Linear(48, 5) |
| `A5_conv3_dropout_bigfc_tanh` | 3 | [8, 16, 24] | [3, 3, 3] | po conv indexech [1] | 0.30 | tanh | Linear(feature_dim, 96) -> tanh -> Linear(96, 5) |

## 5. Vysledky
| Model | Accuracy | Precision | Recall | F1-score | Test loss | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A1_conv1_pool_fc` | 0.5836 | 0.5921 | 0.5842 | 0.5789 | 1.0828 | 8 |
| `A2_conv2_pool_fc` | 0.5636 | 0.5908 | 0.5586 | 0.5514 | 1.1026 | 7 |
| `A3_conv3_pool_fc` | 0.5473 | 0.5650 | 0.5460 | 0.5479 | 1.1162 | 6 |
| `A4_conv2_dropout_fc` | 0.6073 | 0.6120 | 0.6021 | 0.5964 | 1.0340 | 8 |
| `A5_conv3_dropout_bigfc_tanh` | 0.4927 | 0.5064 | 0.4925 | 0.4804 | 1.3196 | 7 |

## 6. Nejlepsi model
Nejlepsi model byl vybran podle nejvyssi test accuracy; pri shode rozhoduje vyssi F1-score a nizsi test loss.
- Nejlepsi architektura: `A4_conv2_dropout_fc`.
- Vysvetleni: tato varianta dosahla nejlepsi kombinace accuracy `0.6073` a F1 `0.5964` pri test loss `1.0340`.
- Ulozeny model: `EXP09/results/best_model.pt`.

## 7. Confusion matrix
Confusion matrix nejlepsiho modelu je ulozena v `EXP09/results/best_confusion_matrix.png`.

![Confusion matrix](EXP09/results/best_confusion_matrix.png)

## 8. Train/validation loss graf
Train a validation loss krivky pro vsechny architektury jsou ulozeny v `EXP09/results/loss_curves.png`.

![Loss curves](EXP09/results/loss_curves.png)

## 9. Ukazka predikci
Ukazky testovacich predikci nejlepsiho modelu jsou ulozeny v `EXP09/results/prediction_examples.png`.

![Prediction examples](EXP09/results/prediction_examples.png)

## 10. Diskuze
Flower dataset je narocnejsi nez male cislove datasety, protoze obsahuje RGB fotografie s vetsimi rozdily v pozadi, meritku, osvetleni a tvaru kvetu. Hlubsi CNN muze zachytit slozitejsi vizualni rysy, ale pri kratkem treninku nemusi vyuzit celou kapacitu.
Pooling zmensuje prostorove rozliseni a pomaha potlacit male posuny objektu. U kvetin je to uzitecne, ale prilis agresivni zmenseni muze odstranit jemne textury okvetnich listku.
Dropout pusobi jako regularizace plne propojene casti. U fotografii muze snizit preuceni na konkretni pozadi, ale pri nizkem poctu epoch muze take zpomalit uceni.
Aktivacni funkce `ReLU` je rychla a stabilni pro vetsinu variant. `Tanh` v A5 meni nelinearitu a muze byt citlivejsi na saturaci, proto je vhodne sledovat nejen accuracy, ale i validation loss.
Tento beh je kratky kontrolni beh nad realnym lokalnim flower datasetem. Vysledky jsou pouzitelne pro porovnani architektur za stejnych podminek, ale pro finalni presnost by bylo vhodne navysit pocet epoch a pripadne pouzit augmentaci.

## 11. Zaver
EXP09 splnuje zadani s omezenim kratkeho treninku: pouziva lokalni flower classification dataset z `/Users/dasky/PycharmProjects/cviceni/dataset/`, pet skutecne odlisnych PyTorch CNN architektur, stejne rozdeleni dat i trenovaci parametry, CSV metriky, porovnavaci graf, confusion matrix, loss krivky, ukazky predikci a ulozeny nejlepsi model.
