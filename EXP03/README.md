# NNUI2 / EXP03

## Hopfield

Implementace diskrétní Hopfieldovy sítě pro NNUI2 (Cvičení 3):
- Hebbovo učení (`W = sum(p p^T)`, nulová diagonála, symetrie)
- energie `E = -0.5 * s^T W s`
- synchronní recall s detekcí fixpointu a 2-cyklu
- toy experiment 3×3
- MNIST experiment (4 uložené vzory + 4 dotazy)
- generování markdown reportu s obrázky a grafy

## Struktura projektu

```text
hopfield/
  __init__.py
  net.py              # HopfieldNet
  experiments.py      # toy + MNIST + generování reportu
scripts/
  run_experiments.py  # CLI entrypoint
tests/
  test_hopfield.py    # rychlé pytest testy
report/
  report.md           # generovaný report
  assets/             # PNG výstupy experimentů
requirements.txt
README.md
```

## Instalace

```bash
pip install -r requirements.txt
```

Volitelné:
- `tensorflow` (preferovaný loader MNIST přes `tensorflow.keras.datasets.mnist`)

Fallback:
- pokud `tensorflow` není dostupné, kód zkusí `scikit-learn` + `fetch_openml("mnist_784")`
- pokud je k dispozici lokální `mnist.npz` (keras formát), loader ho umí načíst offline
- pokud není internet / OpenML je blokované, MNIST část se přeskočí a report se vygeneruje alespoň s toy výsledky

## Spuštění experimentů

```bash
python -m scripts.run_experiments --out report/assets
```

Volitelně se seedem:

```bash
python -m scripts.run_experiments --out report/assets --seed 42
```

Offline varianta s lokálním MNIST cache:

```bash
python -m scripts.run_experiments --out report/assets --mnist-npz /cesta/k/mnist.npz
```

Výstup:
- obrázky a grafy (`.png`) v `report/assets`
- report v `report/report.md`

## Spuštění testů

```bash
pytest -q
```

Testy ověřují:
- symetrii vah a nulovou diagonálu po trénování
- neklesající/nestoupající trend energie v jednoduchém toy recallu
- vybavení uloženého vzoru ze slabě zašuměného vstupu
- shodu s daty ze slajdu „Příklad 1“ (přesná váhová matice a mezistavy synchronního recallu)

## Poznámky k implementaci

- Síť používá bipolární reprezentaci (`{-1, +1}`) interně.
- `preprocess(...)` akceptuje vstupy tvaru `(k,H,W)`, `(k,N)`, `(H,W)`, `(N,)`.
- V `train(...)` se používá ne-normalizované Hebbovo učení `W = Σ p p^T` (kvůli shodě se slajdy / testem „Příklad 1“).
- Recall je synchronní (`s_{t+1} = activation(W @ s_t)`) a loguje:
  - seznam stavů
  - seznam energií
  - metadata (`converged`, `reason`, `iters`)

## Report

Report je generován automaticky skriptem a obsahuje:
- sekce: Cíl, Implementace, Validace podle slajdu (Příklad 1), 3×3 test, MNIST experiment, Zhodnocení
- vložené obrázky (`![](assets/...)`)
- tabulku výsledků MNIST recallu (nebo placeholder, pokud dataset není dostupný)
