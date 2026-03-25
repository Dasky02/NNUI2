# EXP06 – FFNN klasifikace

Experiment pro cvičení 6 obsahuje:

- pět různých topologií FFNN pro klasifikaci
- deset běhů pro každou topologii s různou inicializací
- boxplot testovací chyby nad topologiemi
- confusion matrix a loss curve nejlepšího modelu

Poznámka: zadání cvičení je postavené na PyTorch, ale v tomto prostředí není `torch` dostupný. Ověřená experimentální část je proto realizována přes `sklearn.neural_network.MLPClassifier`, což je stále feedforward neuronová síť. Tato odchylka je přiznaná i v reportu a checklistu.

## Spuštění

```bash
cd /Users/dasky/PycharmProjects/NNUI2/EXP06
python3 run_experiment.py
```

## Výstupy

- `outputs/run_results.csv`
- `outputs/results.json`
- `report/assets/*.png`
- `report/report.md`

