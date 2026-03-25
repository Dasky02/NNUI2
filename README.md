# NNUI2 Experiments

Repozitář obsahuje finální vypracování cvičení 02–06 pro předmět NNUI2 včetně implementací, vygenerovaných výstupů, markdown reportů a submission checklistu.

## Obsah repozitáře

- `EXP02/experiment_02/` – jednoduchý perceptron nad veřejným datasetem Breast Cancer Wisconsin
- `EXP03/` – diskrétní Hopfieldova síť, toy 3x3 experiment a recall na MNIST
- `EXP04/` – Kohonenova samoorganizační mapa pro syntetická data a Iris
- `EXP05/` – FFNN pro regresní aproximaci syntetické úlohy a dat z `Cv5_Data`
- `EXP06/` – FFNN klasifikace nad datasetem Wine, ověřená v tomto prostředí přes `sklearn` fallback
- `common/` – sdílené pomocné utility pro běh experimentů
- `SUBMISSION_CHECKLIST.md` – souhrnná kontrola splnění zadání a rerun příkazy

## Přehled cvičení

### EXP02 – Perceptron
- 10 běhů s různou inicializací vah
- veřejný dataset, boxplot finálních chyb, confusion matrix
- report: [EXP02/experiment_02/report/report.md](EXP02/experiment_02/report/report.md)

### EXP03 – Hopfield
- ověření podle slajdu, toy 3x3, MNIST recall
- logování energií a stavů, report s obrázky
- report: [EXP03/report/report.md](EXP03/report/report.md)

### EXP04 – SOM
- toy experiment se 3 shluky
- Iris ve scénářích exact / insufficient / redundant shape
- report: [EXP04/report/report.md](EXP04/report/report.md)

### EXP05 – FFNN aproximace
- syntetická kontrola objektu FFNN
- více konfigurací pro regresní data s train/validation/test vyhodnocením
- report: [EXP05/report/report.md](EXP05/report/report.md)

### EXP06 – FFNN klasifikace
- 5 topologií, 10 běhů na topologii, boxplot testovacích chyb
- nejlepší model, confusion matrix a loss curve
- report: [EXP06/report/report.md](EXP06/report/report.md)

## Spuštění experimentů

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP02/experiment_02 && python3 main.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP03 && python3 -m scripts.run_experiments --out report/assets --seed 42
cd /Users/dasky/PycharmProjects/NNUI2/EXP04 && python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP05 && python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP06 && python3 run_experiment.py
```

## Testy

```bash
cd /NNUI2/EXP03 && pytest -q
cd /NNUI2/EXP04 && pytest -q
cd /NNUI2/EXP05 && pytest -q
cd /NNUI2/EXP06 && pytest -q
```

## Kde jsou reporty

- `EXP02/experiment_02/report/report.md`
- `EXP03/report/report.md`
- `EXP04/report/report.md`
- `EXP05/report/report.md`
- `EXP06/report/report.md`

## Známé limity prostředí

- V aktuálním prostředí není k dispozici `torch`, proto nebylo možné poctivě ověřit PyTorch variantu EXP06.
- EXP06 je proto explicitně zdokumentován jako ověřený `sklearn.neural_network.MLPClassifier` fallback.
- V aktuálním prostředí není k dispozici `tensorflow`, ale EXP03 bylo možné spustit díky lokální cache `mnist.npz`.

## Poznámka k odevzdání

Repozitář je připraven jako samostatný veřejný Git repozitář pouze pro složku `NNUI2/`. Lokální IDE a virtuální prostředí jsou ignorovány přes `.gitignore`.
