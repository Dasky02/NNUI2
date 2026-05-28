# Přehled experimentů NNUI2

Tento soubor stručně popisuje, co řeší jednotlivá cvičení/experimenty, jaký používají model nebo metodu a s jakými daty pracují. Cílem je rychlá orientace před odevzdáním nebo obhajobou.

## EXP01 - úvod / NumPy

Samostatná složka `EXP01` v projektu aktuálně není nalezena. Podle dostupných materiálů k ní patří `cviceni/NNUI2_01_Cv.pptx` a mělo by jít hlavně o úvodní práci s Pythonem/NumPy bez speciálního datasetu. Pokud je EXP01 vyžadováno v odevzdání, je potřeba jej doplnit nebo jasně označit jako cvičení bez samostatného experimentu.

## EXP02 - perceptron

Experiment ukazuje jednoduchý perceptron pro binární klasifikaci. Používá veřejný dataset `sklearn.datasets.load_breast_cancer`, ze kterého jsou vybrány čtyři příznaky a data jsou standardizována. Perceptron se trénuje opakovaně s různou inicializací vah, ukládá průběh chyby a vyhodnocuje nejlepší běh pomocí accuracy, precision, recall, F1 a confusion matrix.

## EXP03 - Hopfieldova síť

EXP03 implementuje diskrétní Hopfieldovu síť jako asociativní paměť. Nejprve pracuje s malými ručně definovanými vzory 3x3, potom s binarizovanými a zmenšenými vzory z lokálního MNIST souboru. Sleduje se, jestli síť z poškozeného vstupu obnoví uložený vzor, jak se mění energie a jestli skončí ve stabilním bodě nebo oscilaci.

## EXP04 - Kohonenova samoorganizační mapa

Experiment řeší Kohonenovu samoorganizační mapu neboli SOM. Testuje se na syntetických 2D shlucích a na veřejném datasetu Iris, kde jsou vstupy standardizované. Cílem je ukázat, jak počet neuronů a šířka sousedství ovlivňují kvantizační chybu, přiřazení vzorků k neuronům a schopnost mapy zachytit přirozenou strukturu dat.

## EXP05 - dopředná neuronová síť pro regresi

EXP05 implementuje vícevrstvou dopřednou neuronovou síť s jednou skrytou vrstvou a backpropagací. Hlavní data jsou lokální regresní soubory `cviceni/Cv5_Data/train.csv`, `val.csv` a `test.csv`, kde jsou čtyři vstupy a jeden očekávaný výstup. Experiment porovnává několik konfigurací počtu neuronů, learning rate a délky trénování; výsledkem jsou MSE/MAE, křivky učení a graf predikce proti skutečné hodnotě.

## EXP06 - FFNN klasifikace a vliv topologie

EXP06 porovnává pět topologií feedforward neuronové sítě v PyTorch na klasifikační úloze. Používá veřejný dataset `sklearn.datasets.load_wine`, stratifikovaný train/validation/test split a standardizaci příznaků podle trénovací množiny. Každá topologie běží desetkrát s různými seedy, ukládá se testovací chyba každého běhu, boxplot testovacích chyb, confusion matrix a nejlepší model.

Zdrojová složka `cviceni/FFNN II-20260325/` je v reportu uvedena jako referenční materiál k cvičení. Aktuální experiment ji přímo nespouští; implementuje stejné zadání vlastním PyTorch skriptem.

## EXP07 - extrakce příznaků

EXP07 zkoumá, jak různé reprezentace vstupu ovlivní klasifikaci pomocí FFNN. Používá obrazový dataset `sklearn.datasets.load_digits` a porovnává varianty příznaků jako raw pixels, HOG, PCA a HOG+PCA. Nad každou reprezentací se trénuje stejný typ FFNN klasifikátoru, běhy se opakují s různými seedy a výsledek se hodnotí přes accuracy, precision, recall, F1, test error a boxplot.

## EXP08 - CNN a velikost konvolučního kernelu

EXP08 testuje jednu CNN architekturu a mění pouze velikost konvolučního kernelu. Používá dataset `sklearn.datasets.load_digits`, tedy malé grayscale obrázky číslic 8x8, a porovnává kernel size 1x1, 3x3, 5x5, 7x7 a 9x9. Ostatní podmínky zůstávají stejné, aby šel rozdíl ve výsledcích připsat hlavně velikosti kernelu; ukládají se metriky, confusion matrix, feature mapy a nejlepší model.

Zdrojová složka `cviceni/CNN I-20260522/` je uvedena jako referenční materiál k CNN I a konvolučním kernelům. Aktuální EXP08 ji přímo nespouští, ale řeší stejné zadání vlastní PyTorch implementací.

## EXP09 - porovnání CNN architektur

EXP09 porovnává pět různých CNN architektur na lokálním flower classification datasetu `cviceni/dataset/`. Dataset obsahuje třídy `daisy`, `dandelion`, `rose`, `sunflower` a `tulip`; obrázky se načítají ze složek podle názvu třídy, resizeují se na 64x64 a normalizují. Architektury se liší počtem konvolučních vrstev, poolingem, dropoutem, aktivační funkcí a velikostí plně propojené části; výstupem jsou metriky, loss křivky, confusion matrix, ukázky predikcí a uložený nejlepší model.

Aktuální běh je označený jako `PASS_WITH_LIMITATIONS`, protože šlo o krátký kontrolní trénink. Dataset je ale už správně lokální flower dataset, ne původní `sklearn digits`.

## EXP10 - YOLO OBB detekce

EXP10 je připravené pro detekci orientovaných bounding boxů pomocí YOLO26-OBB. Zadání očekává Bricks Detection Dataset s OBB anotacemi a train/validation/test split, ale správný lokální OBB dataset v projektu zatím chybí. Proto experiment negeneruje falešné výsledky; ukládá připravené varianty, příkazy, fallback CSV a report s tím, že plné trénování a test evaluace půjdou spustit až po dodání správného datasetu.

## EXP11 - segmentace Potatoes_seg

EXP11 je sjednocené na lokální segmentační dataset `cviceni/Potatoes_seg/`. Dataset obsahuje `data.yaml`, připravené train/valid/test složky s obrázky a YOLO polygonové anotace v `labels/*.txt` pro jednu třídu `potatoes-`. Experiment převádí polygonové anotace na binární masky a provádí krátký smoke běh malé PyTorch encoder-decoder CNN, která predikuje segmentační masku.

Soubor `cviceni/segmentace.txt` je použit jako referenční YOLO26-seg instrukční materiál. Aktuální stav je `PASS_WITH_LIMITATIONS`, protože vznikly reálné metriky, model a ukázky masek, ale neproběhl dlouhý plný YOLO26-seg trénink.

## EXP12 - lokální LLM přes Ollama

EXP12 ověřuje lokální spuštění LLM přes Ollama a volání HTTP API endpointu `http://localhost:11434/api/generate`. Používá jeden pevný prompt: `Vysvětli rozdíl mezi CNN a FFNN jednoduše pro studenta.` Nad tímto promptem porovnává více konfigurací parametrů generování, například `temperature`, `top_p`, `top_k`, `repeat_penalty` a `num_ctx`.

Pokud Ollama běží, skript ukládá reálné odpovědi do JSON a souhrn do CSV; pokud neběží, má korektní fallback bez vymýšlení odpovědí. Složka `cviceni/ollama_test2/` je v reportu uvedena jako referenční/předchozí materiál pro Ollama API volání, ale aktuální výsledky pochází z vlastního skriptu `EXP12/run_experiment.py`.

## Shrnutí stavu

Většina experimentů má funkční implementaci, report a uložené výstupy. EXP09 a EXP11 byly opraveny na správné lokální datasety a jsou vedené jako `PASS_WITH_LIMITATIONS`, protože byly spuštěny krátké kontrolní běhy. EXP10 zůstává hlavní datový problém, protože pro něj chybí správný OBB detection dataset. EXP01 není v projektu nalezené jako samostatná složka.
