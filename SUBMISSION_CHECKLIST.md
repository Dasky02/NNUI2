# Submission Checklist

## Auditní mapa

| Cvičení | Zadání | Návod / formát | Implementace | Deník | Stav | Poznámka |
| --- | --- | --- | --- | --- | --- | --- |
| 02 | Jednoduchý perceptron na veřejném datasetu + 10 běhů | `navod/Experiment_01/Experiment_01.md` | `EXP02/experiment_02/main.py` | `EXP02/experiment_02/report/report.md` | SPLNĚNO | Uloženy chyby, váhy, boxplot i confusion matrix |
| 03 | Diskrétní Hopfieldova síť + toy 3x3 + MNIST | `navod/Experiment.md` | `EXP03/hopfield/net.py`, `EXP03/scripts/run_experiments.py` | `EXP03/report/report.md` | SPLNĚNO | Ověřeno testy, použita lokální cache `mnist.npz` |
| 04 | Kohonenova SOM + toy shluky + Iris | `navod/Experiment.md` | `EXP04/som.py`, `EXP04/run_experiment.py` | `EXP04/report/report.md` | SPLNĚNO | Vygenerovány křivky kvantizační chyby i přiřazení |
| 05 | FFNN aproximace + syntetická úloha + CSV data | `navod/Experiment.md` | `EXP05/ffnn.py`, `EXP05/run_experiment.py` | `EXP05/report/report.md` | SPLNĚNO | Ověřen backprop, 8 konfigurací s train/val křivkami |
| 06 | FFNN klasifikace + 5 topologií + 10 běhů | `navod/Experiment.md` | `EXP06/run_experiment.py` | `EXP06/report/report.md` | SPLNĚNO S BLOKACÍ | Experiment proběhl reálně, ale kvůli chybějícímu `torch` byl použit ověřený `sklearn` fallback |

## Kde jsou výstupy

- `EXP02/experiment_02/outputs/` obsahuje uložené váhy, průběhy chyb, `run_summary.csv` a `metrics.json`.
- `EXP03/report/assets/` obsahuje toy i MNIST obrázky a grafy energie.
- `EXP04/outputs/results.json` a `EXP04/report/assets/` obsahují metriky a vizualizace SOM.
- `EXP05/outputs/results.json`, `EXP05/outputs/best_model_weights.npz` a `EXP05/report/assets/` obsahují výsledky i grafy všech konfigurací.
- `EXP06/outputs/run_results.csv`, `EXP06/outputs/results.json` a `EXP06/report/assets/` obsahují opakované běhy, boxplot i nejlepší model.

## Co bylo automaticky ověřeno

- `cd /Users/dasky/PycharmProjects/NNUI2/EXP03 && pytest -q` -> `4 passed`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP04 && pytest -q` -> `2 passed`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP05 && pytest -q` -> `2 passed`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP06 && pytest -q` -> `1 passed`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP02/experiment_02 && python3 main.py`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP03 && python3 -m scripts.run_experiments --out report/assets --seed 42`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP04 && python3 run_experiment.py`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP05 && python3 run_experiment.py`
- `cd /Users/dasky/PycharmProjects/NNUI2/EXP06 && python3 run_experiment.py`

## Co vyžaduje ruční kontrolu

- Vizualní kontrola markdown reportů včetně relativních cest k obrázkům.
- Kontrola, zda vyučující akceptuje u cvičení 6 ověřený `sklearn` fallback místo PyTorch implementace.
- Kontrola, zda starší historické soubory v `EXP02/experiment_02/` mimo `outputs/` nemají být před odevzdáním ignorovány.

## Známé limity / blokace prostředí

- Systémový `python3` neobsahuje `torch`, proto není možné v tomto prostředí poctivě ověřit PyTorch variantu cvičení 6.
- Systémový `python3` neobsahuje `tensorflow`, ale Hopfieldův experiment 3 bylo možné spustit díky lokální cache `~/.keras/datasets/mnist.npz`.
- Kořen `/Users/dasky/PycharmProjects` není Git repozitář, proto je soupis změn veden jen v tomto checklistu a finálním shrnutí.

## Přesné příkazy pro spuštění experimentů

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP02/experiment_02 && python3 main.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP03 && python3 -m scripts.run_experiments --out report/assets --seed 42
cd /Users/dasky/PycharmProjects/NNUI2/EXP04 && python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP05 && python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP06 && python3 run_experiment.py
```

## Přesné příkazy pro regeneraci reportů

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP02/experiment_02 && python3 main.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP03 && python3 -m scripts.run_experiments --out report/assets --seed 42
cd /Users/dasky/PycharmProjects/NNUI2/EXP04 && python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP05 && python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP06 && python3 run_experiment.py
```
