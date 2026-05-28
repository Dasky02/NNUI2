# NNUI2 Experiments

Repozitář obsahuje vypracování a audit cvičení podle PDF zadání ve složce `../cviceni/`.

## Obsah repozitáře

- `EXP02/experiment_02/` – starší perceptron mimo aktuální PDF audit
- `EXP03/` – diskrétní Hopfieldova síť, toy 3x3 a MNIST
- `EXP04/` – Kohonenova samoorganizační mapa, toy shluky a Iris
- `EXP05/` – starší FFNN aproximace mimo aktuální PDF audit
- `EXP06/` – FFNN klasifikace, skutečná PyTorch FFNN nad datasetem Wine
- `EXP07/` – extrakce příznaků, skutečná PyTorch FFNN
- `EXP08/` – vliv velikosti kernelu, skutečná PyTorch CNN
- `EXP09/` – vliv architektury CNN, skutečná PyTorch CNN
- `EXP10/` – ready-to-run YOLO26 OBB skript, blokováno prostředím/datasetem
- `EXP11/` – ready-to-run YOLO26 segmentační skript, blokováno prostředím/datasetem
- `EXP12/` – lokální LLM přes Ollama API, porovnání parametrů generování
- `common/` – sdílené pomocné utility pro běh experimentů

## Spuštění ověřených experimentů

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP03 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m scripts.run_experiments --out report/assets --seed 42
cd /Users/dasky/PycharmProjects/NNUI2/EXP04 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP06 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP07 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP08 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP09 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
cd /Users/dasky/PycharmProjects/NNUI2/EXP12 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py
```

## YOLO ready-to-run příkazy

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP10 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py --data /path/to/bricks_obb/data.yaml --run
cd /Users/dasky/PycharmProjects/NNUI2/EXP11 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run_experiment.py --data /path/to/bricks_seg_topbricks/data.yaml --run
```

## Testy

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP03 && pytest -q
cd /Users/dasky/PycharmProjects/NNUI2/EXP04 && pytest -q
cd /Users/dasky/PycharmProjects/NNUI2/EXP05 && pytest -q
cd /Users/dasky/PycharmProjects/NNUI2/EXP06 && pytest -q
cd /Users/dasky/PycharmProjects/NNUI2/EXP07 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q
cd /Users/dasky/PycharmProjects/NNUI2/EXP08 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q
cd /Users/dasky/PycharmProjects/NNUI2/EXP09 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q
cd /Users/dasky/PycharmProjects/NNUI2/EXP12 && /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest -q
```

## Známé limity prostředí

- `tensorflow` není dostupný.
- `torch`, `ultralytics` a `ollama` jsou dostupné v ověřeném prostředí; systémový Python `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` byl použit pro EXP06–EXP12.
- Cvičení 10–11 jsou připravena ke spuštění, ale chybí správný Bricks Detection Dataset ze Zenodo.
