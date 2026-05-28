# NNUI2 - Cviceni 7: Extrakce priznaku

## 1. Cil experimentu
Cilem je overit vliv ruznych reprezentaci vstupu a extrakce priznaku na klasifikaci pomoci stejne FFNN. Porovnavaji se raw pixely, HOG, PCA z raw pixelu a PCA z HOG priznaku.

## 2. Dataset
- Nazev datasetu: `sklearn.datasets.load_digits`.
- Typ dat: `obrazovy dataset 8x8 grayscale cislic`.
- Pocet vzorku: `1797`.
- Pocet trid: `10`.
- Rozdeleni train/val/test: `1077/360/360` = `60/20/20 stratified`.

## 3. Preprocessing
- Obrazy 8x8 jsou prevedeny na `float32` a normalizovany z rozsahu 0-16 do rozsahu 0-1.
- Pro raw pixely se obraz flattenuje na 64D vektor.
- HOG-like varianta pocita gradienty a histogramy orientaci v bunkach.
- PCA varianty jsou fitovane pouze na trenovaci casti, aby nedochazelo k data leakage.
- Pred trenovanim FFNN je kazda varianta skálovana pomoci `StandardScaler` fitovaneho pouze na train splitu.

## 4. Varianty priznaku
| Varianta | Metoda extrakce | Dimenze | Parametry |
| --- | --- | ---: | --- |
| `raw_pixels_64` | flatten raw pixels | 64 | 8x8 obraz serializovany na 64 hodnot |
| `hog_4x4_9bins` | HOG-like histogram orientovanych gradientu | 144 | 4x4 bunky, 9 binu na bunku |
| `pca_16_from_raw` | PCA z raw pixelu | 16 | 16 komponent, fit pouze na train splitu |
| `hog_pca_24` | PCA z HOG priznaku | 24 | HOG 144D redukovany na 24 komponent, fit pouze na train splitu |

## 5. Model
Pro vsechny varianty je pouzita stejna PyTorch FFNN (`torch.nn.Module`): vstupni vrstva podle dimenze priznaku, skryte vrstvy `(48, 24)`, aktivace `ReLU`, vystupni vrstva pro 10 trid, `CrossEntropyLoss`, optimizer `Adam`, learning rate `0.003`, batch size `32`, max `120` epoch s early stopping podle validacni accuracy.

| Varianta | Vstupni dimenze | Hidden vrstvy | Aktivace | Optimizer | Epochs |
| --- | ---: | --- | --- | --- | ---: |
| `raw_pixels_64` | 64 | (48, 24) | ReLU | Adam, lr=0.003 | 120 |
| `hog_4x4_9bins` | 144 | (48, 24) | ReLU | Adam, lr=0.003 | 120 |
| `pca_16_from_raw` | 16 | (48, 24) | ReLU | Adam, lr=0.003 | 120 |
| `hog_pca_24` | 24 | (48, 24) | ReLU | Adam, lr=0.003 | 120 |

## 6. Opakovani
Kazda varianta byla spustena `5x` s ruznymi reprodukovatelnymi seedy. Celkem probehlo `20` trenovani. Seedy a testovaci chyby jsou v `EXP07/results/metrics.csv`.

## 7. Vysledky
| Varianta | Dimenze | Accuracy mean±std | Precision mean±std | Recall mean±std | F1 mean±std | Test error mean±std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hog_4x4_9bins` | 144 | 0.9556±0.0065 | 0.9567±0.0056 | 0.9553±0.0064 | 0.9553±0.0062 | 0.0444±0.0065 |
| `hog_pca_24` | 24 | 0.9417±0.0068 | 0.9421±0.0061 | 0.9411±0.0069 | 0.9403±0.0070 | 0.0583±0.0068 |
| `pca_16_from_raw` | 16 | 0.9678±0.0064 | 0.9675±0.0065 | 0.9674±0.0064 | 0.9671±0.0066 | 0.0322±0.0064 |
| `raw_pixels_64` | 64 | 0.9628±0.0025 | 0.9632±0.0026 | 0.9625±0.0025 | 0.9624±0.0025 | 0.0372±0.0025 |

## 8. Boxplot
Boxplot porovnava testovaci chybu pro jednotlive varianty priznaku: `EXP07/results/feature_boxplot.png`.

![Boxplot](EXP07/results/feature_boxplot.png)

## 9. Confusion matrix nejlepsiho modelu
Confusion matrix nejlepsiho behu je ulozena v `EXP07/results/confusion_matrix_best.png`.

![Confusion matrix](EXP07/results/confusion_matrix_best.png)

## 10. Nejlepsi model
- Nejlepsi priznaky podle jednotliveho behu: `pca_16_from_raw`.
- Nejlepsi beh: run `5`, seed `7305`.
- Metriky: accuracy `0.9750`, precision `0.9749`, recall `0.9747`, F1 `0.9746`, test error `0.0250`.
- Ulozeny model: `EXP07/results/best_model.pt`.

```text
              precision    recall  f1-score   support

           0       1.00      0.97      0.99        36
           1       0.92      0.92      0.92        36
           2       0.97      1.00      0.99        35
           3       1.00      1.00      1.00        37
           4       0.95      1.00      0.97        36
           5       0.97      1.00      0.99        37
           6       1.00      1.00      1.00        36
           7       1.00      1.00      1.00        36
           8       0.94      0.89      0.91        35
           9       1.00      0.97      0.99        36

    accuracy                           0.97       360
   macro avg       0.97      0.97      0.97       360
weighted avg       0.98      0.97      0.97       360

```

## 11. Diskuze
Nejlepsi prumernou testovaci chybu dosahla varianta `pca_16_from_raw`. To ukazuje, ze pro tento dataset neni rozhodujici pouze nejlepsi jeden beh, ale i stabilita mezi inicializacemi.
Raw pixely zachovavaji vsechny hodnoty obrazu a u malych 8x8 cislic mohou byt velmi silne, protoze FFNN primo vidi plny raster. Jejich nevyhodou je vetsi vstupni dimenze a slabsi vestavena invariantnost vuci posunum nebo tvarovym zmenam.
HOG priznaky explicitne popisuji smer hran, coz je pro cislice prirozene. Na velmi malych obrazech 8x8 je ale gradientova informace hruba, a proto HOG nemusi vzdy prekonat raw pixely.
PCA snizuje dimenzi a odstranuje cast sumu nebo redundance. Nizsi dimenze muze zlepsit stabilitu a zmensit pocet parametru prvni vrstvy, ale prilis agresivni redukce muze ztratit jemne rozdily mezi podobnymi cislicemi.
Nejvyssi dimenzi ma `hog_4x4_9bins` (144D), nejnizsi dimenzi ma `pca_16_from_raw` (16D). Porovnani ukazuje, ze vetsi dimenze automaticky neznamena lepsi vysledek; dulezita je informacni hodnota reprezentace pro konkretni dataset.
Stabilita mezi peti behy je dana jak reprezentaci, tak inicializaci FFNN. Varianta s nizkou prumernou chybou a malou smerodatnou odchylkou je prakticky vhodnejsi nez varianta s jednim vyjimecnym behem a velkym rozptylem.

## 12. Zaver
EXP07 splnuje zadani: pouziva jeden verejny obrazovy dataset, stratifikovane train/validation/test rozdeleni, preprocessing, ctyri skutecne varianty priznaku, PyTorch FFNN, pet behu na variantu, ulozene testovaci chyby, boxplot, confusion matrix a diskuzi vlivu typu i dimenze priznaku.
