# Audit zdroju a datasetu pro NNUI2

Datum auditu: 2026-05-24

Rozsah auditu: `/Users/dasky/PycharmProjects/cviceni` a existujici experimenty `EXP01` az `EXP12` v projektu `NNUI2`.

Audit puvodne kontroloval prirazeni zdroju bez dlouheho trenovani. EXP10 byl pozdeji doplnen o realny Bricks OBB dataset a kratky YOLO OBB smoke beh.

| Experiment | Ocekavany zdroj | Nalezeny zdroj | Typ ulohy | Stav | Problem | Doporuceni |
|---|---|---|---|---|---|---|
| EXP01 | `NNUI2_01_Cv.pptx` | Slozka `EXP01` v projektu nenalezena. | Uvod, NumPy | NOT_FOUND | Cviceni neni v projektu pripraveno jako samostatny experiment. | Vytvorit `EXP01` nebo v odevzdani jasne uvest, ze EXP01 nema samostatny behovy experiment. |
| EXP02 | `NNUI2_02_Cv.pptx` | `EXP02/experiment_02`, report `experiment_02.md`; perceptron nad Breast Cancer datasetem. | Perceptron | OK | Zdrojovy PPT neni v reportu explicitne zminen. | Doplneni odkazu na `NNUI2_02_Cv.pptx` by zlepsilo dohledatelnost. |
| EXP03 | `NNUI2_03_Cv-3.pdf` | `EXP03`, report `experiment_03.md`; Hopfieldova sit, male vzory/MNIST varianta. | Hopfieldova sit | OK | Nebyla nalezena zjevna zamena datasetu. | Ponechat; pripadne v reportu zminit konkretni PDF jako zdroj zadani. |
| EXP04 | `NNUI2_04_Cv.pdf` + `Kohonen.py` | `EXP04`, report `experiment_04.md`; SOM/Kohonen, Iris a synteticke vzory. | Kohonenova mapa / SOM | OK | `Kohonen.py` je dostupny zdrojovy skript, ale neni zjevne primo pouzit nebo citovan. | V reportu doplnit vazbu na `Kohonen.py`, pokud byl pouzit jako vzor. |
| EXP05 | `NNUI2_05_Cv.pptx` + `Cv5_Data/` + `Dopredna neuronova sit.py` | `EXP05`, `run_experiment.py` nacita `cviceni/Cv5_Data/train.csv`, `val.csv`, `test.csv`; data maji `x1,x2,x3,x4,y_real`. | Dopredna sit I, regrese | OK | Nebyla nalezena klasifikacni data misto regresnich dat. | Ponechat; volitelne zminit `Dopredna neuronova sit.py` jako puvodni material. |
| EXP06 | `NNUI2_06_Cv.pdf` + `FFNN II-20260325/` | `EXP06`, `run_experiment.py` pouziva Wine dataset a PyTorch FFNN; 5 topologii, 10 opakovani. Report doplnen o referencni zdroj `FFNN II-20260325/`. | FFNN klasifikace / topologie | OK | Zdrojova slozka je uvedena jako referencni material, ne jako primo importovany/spusteny kod. | Bez dalsi dokumentacni opravy; pripadne jen navysit provazanost s PDF podle pozadavku vyucujiciho. |
| EXP07 | `NNUI2_07_Cv.pdf` | `EXP07`, `run_experiment.py` pouziva `sklearn` digits a varianty raw pixels, HOG, PCA, HOG+PCA. | Extrakce priznaku + FFNN | OK | Nebyla nalezena zmena pouze modelu bez zmeny priznaku. | Ponechat. |
| EXP08 | `NNUI2_08_Cv.pdf` + `CNN I-20260522/` | `EXP08`, `run_experiment.py` pouziva `sklearn` digits; meni 1x1, 3x3, 5x5, 7x7, 9x9 kernel. Report doplnen o referencni zdroj `CNN I-20260522/`. | CNN I, vliv kernelu | OK | Zdrojova slozka je uvedena jako referencni material, ne jako primo importovany/spusteny kod. | Bez dalsi dokumentacni opravy; pro vetsi shodu lze pozdeji navazat konkretni priklady z `CNN I-20260522/`. |
| EXP09 | `NNUI2_09 Cv.pdf` + `dataset/` + `CCN II.py` | `EXP09`, `run_experiment.py` pouziva lokalni `cviceni/dataset/` s tridami `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`; provedeno 5 CNN architektur. | CNN II, porovnani architektur | OK | Experiment byl opraven z `sklearn` digits na lokalni flower classification dataset. Beh je kratky a v reportu oznaceny jako `PASS_WITH_LIMITATIONS`. | Pro finalni presnost lze navysit pocet epoch nebo pridat augmentaci; zdroj `CCN II.py` lze doplnit jako historicky material. |
| EXP10 | `NNUI2_10_Cv.pdf` + Bricks Detection Dataset ze Zenodo `10.5281/zenodo.18952529` | `EXP10` pouziva OBB vetev `/Users/dasky/PycharmProjects/cviceni/datasets/Bricks_Detection_Dataset/OBB/data_all.yaml`; train/val/test = 83/20/30 obrazku, 3 tridy, labely maji 9 hodnot. `dataset/` s kvetinami ani segmentation cast nejsou pouzity. | YOLO OBB detekce | OK / PASS_WITH_LIMITATIONS | Probehlo 5 variant jako 1-epoch smoke na CPU. YOLO26 vahy nebyly v Ultralytics dostupne, proto skript realne pouzil kompatibilni `yolo11n-obb.pt`/`yolo11s-obb.pt` a zapsal fallback do reportu. | Pro finalni presnost spustit delsi beh, napr. `--epochs 10`; kratky smoke je validni jen jako omezeny overovaci beh. |
| EXP11 | `NNUI2_11_Cv.pdf` + `Potatoes_seg/` + `segmentace.txt` | `EXP11` pouziva lokalni `Potatoes_seg/` jako hlavni dataset; `segmentace.txt` je zohlednen jako YOLO26-seg instrukcni material. Dataset ma `data.yaml`, train/valid/test obrazky a YOLO polygonove anotace. | Segmentace | OK | Proveden kratky PyTorch CNN smoke beh, proto je report oznacen jako `PASS_WITH_LIMITATIONS`; neprobehl dlouhy plny YOLO26-seg trenink. | Pro finalni kvalitu segmentace navysit pocet epoch nebo spustit YOLO26-seg podle `segmentace.txt` nad stejnym `data.yaml`. |
| EXP12 | `NNUI2_12_Cv.pdf` + `ollama_test2/` | `EXP12`, `run_experiment.py` vola Ollama API, uklada `ollama_outputs.json` a `summary.csv`. Report doplnen o referencni zdroj `ollama_test2/`. | Lokalni LLM / Ollama | OK | `ollama_test2/` je uveden jako referencni/predchozi material; aktualni vysledky pochazeji z vlastniho `EXP12/run_experiment.py`. | Bez dalsi dokumentacni opravy. |

## Dulezita zjisteni

- `dataset/` obsahuje klasifikacni flower dataset (`daisy`, `dandelion`, `rose`, `sunflower`, `tulip`) a podle dostupnych zdroju patri k EXP09.
- `Potatoes_seg/` obsahuje YOLO segmentacni strukturu `train/valid/test` s `images` a `labels`; po oprave je pouzit jako hlavni dataset EXP11.
- `Cv5_Data/` patri k EXP05. Soubory `train.csv`, `val.csv`, `test.csv` maji regresni strukturu se 4 vstupy a jednim vystupem.
- `FFNN II-20260325/` patri k EXP06 jako zdrojovy material pro FFNN II.
- `CNN I-20260522/` patri k EXP08 jako zdrojovy material pro CNN I a konvolucni kernely.
- `ollama_test2/` patri k EXP12 jako zdrojovy material pro Ollama API volani.
- EXP10 ma doplnenou OBB cast Bricks Detection Datasetu ze Zenodo. Labely byly overeny jako OBB format s 9 hodnotami na radek. Stav je `PASS_WITH_LIMITATIONS`, protoze probehl jen smoke beh a misto nedostupnych YOLO26 vah byl pouzit kompatibilni YOLO11 OBB fallback.

## Rizikove zameny datasetu

| Kontrola | Vysledek | Poznamka |
|---|---|---|
| Flower dataset pouzity pro EXP10 | NE | Nebyl nalezen odkaz na `cviceni/dataset/` v EXP10. |
| `Potatoes_seg/` pouzit pro EXP09 | NE | EXP09 aktualne pouziva lokalni flower dataset z `cviceni/dataset/`. |
| `Cv5_Data/` pouzit mimo EXP05 | NE | Odkazy byly nalezeny pouze v EXP05. |
| Segmentacni dataset pouzit pro klasifikaci bez vysvetleni | NE | Nebylo nalezeno. |
| Klasifikacni dataset pouzit pro detekci nebo segmentaci | NE | Flower dataset neni pouzit pro EXP10/EXP11. |
| Report oznacuje experiment jako PASS, ale chybi spravna data | NE | EXP10 uz pouziva realny Bricks OBB dataset; je oznacen jako `PASS_WITH_LIMITATIONS` kvuli 1-epoch smoke behu a YOLO11 fallbacku. EXP11 je take veden jako `PASS_WITH_LIMITATIONS`, protoze probehl jen kratky smoke beh. |

## Provedene bezpecne kontroly

- Existence hlavni slozky `/Users/dasky/PycharmProjects/cviceni`: OK.
- Struktura `dataset/`: OK, pet trid kvetin.
- Struktura `Potatoes_seg/`: OK, `train/valid/test` se slozkami `images` a `labels`, `data.yaml` nalezen.
- Struktura `Cv5_Data/`: OK, `train.csv`, `val.csv`, `test.csv` nalezeny.
- Struktura `FFNN II-20260325/`: OK, tri Python skripty nalezeny.
- Struktura `CNN I-20260522/`: OK, tri Python skripty nalezeny.
- Struktura `ollama_test2/`: OK, dva Python skripty nalezeny.
- Struktura Bricks OBB datasetu pro EXP10: OK, `train/val/test` obsahuje `images` a `labels`; vsechny neprazdne label radky maji 9 hodnot.
- Python syntax check pro dostupne `.py` soubory v projektu `NNUI2`: OK.
- Python syntax check pro `.py` soubory v `/cviceni`: OK.
- `pytest` z korene projektu: FAIL kvuli kolizi modulu `run_experiment.py` mezi vice EXP slozkami pri globalnim test discovery.
- `pytest` po jednotlivych experimentech EXP03 az EXP12: OK.

## Shrnuty stav

Celkovy stav auditu: PARTIAL.

Hlavni duvod: EXP01 neni v projektu nalezeno jako samostatny experiment. EXP10 je po doplneni Bricks OBB datasetu ve stavu `PASS_WITH_LIMITATIONS`.
