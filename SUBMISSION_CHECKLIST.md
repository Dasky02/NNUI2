# Submission checklist NNUI2

Datum auditu: 2026-05-24

Tento checklist shrnuje stav zdroju, dat a souladu se zadanim pro EXP01 az EXP12. EXP10 byl pozdeji doplnen o realny Bricks OBB dataset a kratky smoke beh.

| Experiment | Pouzite zdroje | Stav dat | Odpovida zadani | Co jeste chybi |
|---|---|---|---|---|
| EXP01 | Ocekavano `cviceni/NNUI2_01_Cv.pptx`; v projektu nenalezeno `EXP01`. | Bez datasetu / nenalezeno. | NE / NOT_FOUND | Vytvorit EXP01 nebo jasne oznacit, ze cviceni nema samostatny experiment. |
| EXP02 | `EXP02/experiment_02`, `experiment_02.md`; ocekavano `NNUI2_02_Cv.pptx`. | Public dataset Breast Cancer. | ANO s drobnou poznamkou | Dopsat zdroj `NNUI2_02_Cv.pptx` do reportu, pokud je vyzadovana stopa na material. |
| EXP03 | `EXP03`, `experiment_03.md`; ocekavano `NNUI2_03_Cv-3.pdf`. | Male vzory / MNIST varianta podle Hopfield experimentu. | ANO | Pripadne doplnit explicitni odkaz na PDF zdroj. |
| EXP04 | `EXP04`, `experiment_04.md`; ocekavano `NNUI2_04_Cv.pdf` + `Kohonen.py`. | Iris / synteticke SOM vzory. | ANO | Dopsat, zda a jak byl pouzit `Kohonen.py`. |
| EXP05 | `EXP05/run_experiment.py`, `cviceni/Cv5_Data/`, report; ocekavano take `NNUI2_05_Cv.pptx` + `Dopredna neuronova sit.py`. | OK, regresni data `x1,x2,x3,x4,y_real`. | ANO | Volitelne doplnit puvodni skript jako zdroj. |
| EXP06 | `EXP06/run_experiment.py`, Wine dataset; report zminuje `NNUI2_06_Cv.pdf` + referencni slozku `FFNN II-20260325/`. | OK pro verejny klasifikacni dataset. | ANO | Dokumentacni varovani vyreseno; zdrojova slozka je uvedena jako referencni, ne primo spoustena. |
| EXP07 | `EXP07/run_experiment.py`, `sklearn` digits; ocekavano `NNUI2_07_Cv.pdf`. | OK, jeden dataset a vice reprezentaci priznaku. | ANO | Nic zasadniho. |
| EXP08 | `EXP08/run_experiment.py`, `sklearn` digits; report zminuje `NNUI2_08_Cv.pdf` + referencni slozku `CNN I-20260522/`. | OK pro maly obrazovy dataset. | ANO | Dokumentacni varovani vyreseno; zdrojova slozka je uvedena jako referencni, ne primo spoustena. |
| EXP09 | `EXP09/run_experiment.py`, lokalni `cviceni/dataset/` s kvetinami; ocekavano `NNUI2_09 Cv.pdf` + `cviceni/dataset/` + `CCN II.py`. | OK, flower classification dataset, 5 trid, 2746 obrazku. | ANO / PASS_WITH_LIMITATIONS | Pro finalni presnost lze navysit pocet epoch; volitelne doplnit `CCN II.py` jako puvodni zdroj. |
| EXP10 | `EXP10/run_experiment.py`, `NNUI2_10_Cv.pdf`, Bricks Detection Dataset ze Zenodo `10.5281/zenodo.18952529`, OBB vetev `/Users/dasky/PycharmProjects/cviceni/datasets/Bricks_Detection_Dataset/OBB/data_all.yaml`. | OK, OBB dataset, 3 tridy, train/val/test = 83/20/30 obrazku, labely maji 9 hodnot. | ANO / PASS_WITH_LIMITATIONS | Spustit delsi beh, napr. `--epochs 10`, pokud je potreba finalni presnost; aktualne probehl 1-epoch smoke na CPU s YOLO11 OBB fallbackem misto nedostupnych YOLO26 vah. |
| EXP11 | `EXP11/run_experiment.py`, lokalni `cviceni/Potatoes_seg/` + `segmentace.txt`. | OK, YOLO polygonovy segmentacni dataset, 1 trida, 300 obrazku. | ANO / PASS_WITH_LIMITATIONS | Pro finalni kvalitu navysit pocet epoch nebo spustit plny YOLO26-seg trenink podle `segmentace.txt`. |
| EXP12 | `EXP12/run_experiment.py`; report zminuje `NNUI2_12_Cv.pdf` + referencni slozku `ollama_test2/`. | OK, bez datasetu; uklada API vystupy podle dostupnosti Ollama. | ANO | Dokumentacni varovani vyreseno; `ollama_test2/` je uveden jako referencni/predchozi material, ne jako zdroj aktualnich vysledku. |

## Celkovy stav

PARTIAL

## Nejdulezitejsi opravy pred odevzdanim

1. EXP01: doplnit chybejici experiment nebo jej explicitne oznacit jako cviceni bez behove casti.
2. EXP10: pro finalni kvalitu navysit pocet epoch; dataset a smoke pipeline jsou uz funkcni.

## Overeni

- Syntax check projektu `NNUI2`: OK.
- Syntax check zdrojove slozky `/cviceni`: OK.
- Testy EXP03 az EXP12 spustene po jednotlivych slozkach: OK.
- Globalni `pytest` z korene projektu: FAIL kvuli import kolizi stejnojmennych `run_experiment.py`; doporuceni je spoustet testy po experimentech nebo upravit test layout/importy.
- EXP10 syntax check: OK.
- EXP10 testy: `pytest EXP10/tests -q` OK, 3 testy.
- EXP10 smoke: OK, 5/5 variant completed nad realnym OBB datasetem.
