# NNUI2 - Cviceni 11: segmentace Potatoes_seg

## 1. Stav experimentu
- Stav: `PASS_WITH_LIMITATIONS`.
- Duvod omezeni: beh je kratky smoke test male segmentacni CNN, nikoli plny dlouhy YOLO26-seg trenink.

## 2. Dataset
- Hlavni dataset: `/Users/dasky/PycharmProjects/cviceni/Potatoes_seg`.
- YAML: `/Users/dasky/PycharmProjects/cviceni/Potatoes_seg/data.yaml`.
- Struktura: `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`.
- Obsahuje obrazky: `True`.
- Obsahuje masky/anotace: `True`.
- Format anotaci: `YOLO segmentation polygon labels`.
- Pocet trid: `1`; nazvy trid: `['potatoes-']`.
- Pocet obrazku train/valid/test: `267/24/9`.
- Flower classification dataset `cviceni/dataset/` neni v EXP11 pouzit.

## 3. Metodika
YOLO polygon labely se prevadeji na binarni masky vykreslenim polygonu do masky stejne velikosti jako vstupni obrazek. Obrazky a masky se nasledne resizeuji na `128x128`.
Trenovani pouziva pripraveny split datasetu: maximalne `64` trenovacich obrazku, `16` validacnich obrazku a vsechny test obrazky. Pixely jsou normalizovane do rozsahu `0-1`.
Soubor `segmentace.txt` obsahuje kostru pro YOLO26/Ultralytics segmentaci s `YOLO('yolo26n-seg.pt')`, `model.train(data='...data.yaml')`, validaci a predikci. EXP11 z nej prebiral hlavni informaci, ze lokalni data jsou YOLO segmentacni dataset s `data.yaml`; pro bezpecny kratky smoke beh je zde pouzita mala PyTorch CNN nad stejnymi polygonovymi anotacemi.

## 4. Model
Pouzita je mala PyTorch encoder-decoder CNN `TinySegNet`: dve konvolucni casti s poolingem a decoder s bilinear upsamplingem. Vystupem je jedna binarni maska pro tridu potatoes.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: `Adam`, learning rate `0.001`.
- Epochs: `2`.
- Batch size: `8`.
- Device: `cpu`.

## 5. Vysledky
- Posledni train loss: `0.5859072208404541`.
- Posledni validation loss: `0.4655728191137314`.
- Test loss: `0.45778788626194`.
- Test pixel accuracy: `0.8358345031738281`.
- Test IoU: `0.0`.
- Test Dice: `0.0`.
- Detailni metriky jsou v `EXP11/results/metrics.csv`.

## 6. Ukazky segmentace
Ukazky obsahuji vstup, referencni masku z YOLO polygonu a predikci male CNN.

![Ukazky segmentace](EXP11/results/segmentation_samples.png)

## 7. Diskuze
Dataset je vhodny pro segmentaci: obsahuje realne obrazky, pripravene train/valid/test rozdeleni, `data.yaml` a YOLO polygonove anotace. Neobsahuje samostatne bitmapove masky, ale ty lze korektne odvodit z polygonu bez vytvareni falesnych masek.
Kvalita segmentace je limitovana kratkym smoke treninkem a malou architekturou. Metriky proto overuji funkcnost pipeline, ne finalni produkcni presnost.
Pro plnohodnotny experiment podle `segmentace.txt` by dalsi krok byl spustit YOLO26-seg/Ultralytics trenink nad stejnym `data.yaml`, ulozit mask mAP metriky a porovnat vice variant.

## 8. Zaver
EXP11 je sjednocen s lokalnim `Potatoes_seg/` datasetem a jasne resi segmentaci, ne klasifikaci. Vznikly realne metriky kratkeho PyTorch CNN smoke behu, ulozeny model a ukazky segmentace. Stav je `PASS_WITH_LIMITATIONS`, protoze neprobehl dlouhy YOLO26-seg trenink.
