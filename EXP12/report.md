# NNUI2 - Cviceni 12: Lokalni LLM pomoci Ollama

## 1. Cil
Cilem je overit lokalni spusteni LLM pres Ollama, zavolat model pres HTTP API a porovnat vliv parametru generovani na odpoved na jeden pevny prompt.

## Zdrojove materialy
- Zadani/material cviceni: `/Users/dasky/PycharmProjects/cviceni/NNUI2_12_Cv.pdf`.
- Dostupna zdrojova slozka: `/Users/dasky/PycharmProjects/cviceni/ollama_test2/`.
- Poznamka k pouziti: skripty v `ollama_test2/` nebyly primo importovany ani spousteny aktualnim experimentem. Soucasny EXP12 pouziva vlastni skript `EXP12/run_experiment.py`, ktery vola stejny Ollama endpoint `http://localhost:11434/api/generate` a testuje vice konfiguraci generovani. `ollama_test2/` je tedy referencni/predchozi material pro API volani, ne primy zdroj vysledku v tomto reportu.

## 2. Prostredi
- OS: `macOS-26.2-arm64-arm-64bit`.
- Python: `3.12.5`.
- Ollama: `ollama version is 0.22.1`.
- Pouzity model: `gemma3:1b`.
- CPU/GPU/RAM: CPU `arm`, CPU count `8`, RAM `8.0 GB`; GPU nebylo pres skript samostatne zjistovano.
- Dostupnost API: `available`.

## 3. Spusteni Ollama
```bash
ollama serve
ollama pull gemma3:1b
```

## 4. API volani
- Endpoint: `http://localhost:11434/api/generate`.
- Python request pouzity ve skriptu:
```python
payload = {
    "model": "gemma3:1b",
    "prompt": PROMPT,
    "stream": False,
    "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_ctx": 2048},
}
```

## 5. Testovany prompt
`Vysvětli rozdíl mezi CNN a FFNN jednoduše pro studenta.`

## 6. Testovane varianty parametru
| Varianta | temperature | top_p | top_k | repeat_penalty | num_ctx |
| --- | ---: | ---: | ---: | ---: | ---: |
| `V1` | 0.2 | 0.9 | 40 | 1.1 | 2048 |
| `V2` | 0.8 | 0.9 | 40 | 1.1 | 2048 |
| `V3` | 1.2 | 1.0 | 100 | 1.05 | 2048 |
| `V4` | 0.2 | 0.8 | 20 | 1.2 | 2048 |
| `V5` | 0.7 | 0.95 | 80 | 1.1 | 8192 |

## 7. Vysledky
### V1
- Stav: `completed`.
- Delka odpovedi: `309` slov, `1913` znaku.
- Cas generovani: `15.359` s.
- Subjektivni kvalita: slabsi - chybi cast klicovych pojmu.
- Opakovani: opak.
- Kreativita: nevyhodnoceno.
- Konzistence: nizka.

```text
Představ si, že se učíš rozpoznávat kočky a psy. Oba typy neuronových sítí, CNN a FFNN, se snaží najít vzory v datech, ale fungují trochu jinak. Zde je rozdíl v jednodušším jazyce:

**FFNN (Feedforward Neural Network) - Jako jednoduchý "přepínač"**

*   **Jak to funguje:** Představ si, že máš jednoduchý "přepínač", který se na každou část dat podívá a rozhodne, zda je to kočka nebo pes.  Vypočítá jednoduchou pravidla, která určuje, jestli je daný vzor kočka nebo pes.
*   **Jak se učí:**  FFNN se učí, když se na ní dá podávat *velká* množství dat.  Je to jako učení se na příkladech.
*   **Příklad:** Představ si, že se učíš rozpoznávat obrázky koček a psů.  FFNN se naučí, že kočky mají specifické rysy (např. větší uši) a psi mají jiné.

**CNN (Convolutional Neural Network) - Jako "vytvářič"**

*   **Jak to funguje:** CNN se zaměřuje na *vztahy* v datech, nikoli na jednoduché pravidla.  Místo toho používá "vytvářiče" (convolutional layers) - to jsou malé "vytvářiče", které se pohybují po obrázku.
*   **Jak se učí:** CNN se učí, že kočky a psi mají *různé vzory* v obrázcích.  Vytvářiče "vytvoří" "vytvářiče" (např. "vytvářiče pro uši") které se zaměřují na *konkrétní* rysy v obrázku.  Pak se CNN učí, jak tyto rysy kombinovat a rozpoznat vzory.
*   **Příklad:** Představ si, že se učíš rozpoznávat obrázky koček. CNN se naučí, že kočky mají specifický tvar obličeje, kůží a vlasy.  Vytvářiče se budou pohybovat po obrázku a hledat tyto rysy.

**Zjednodušeně:**

*   **FFNN:**  "Převede" data na pravidla.
*   **CNN:** "Vytvoří" vzory v datech, které se pak používají k rozpoznání.

**Který je lepší?**

CNN jsou obvykle lepší pro rozpoznávání obrázků, protože se specializují na detekování *vztahů* v datech. FFNN jsou užitečné pro jiné typy dat, kde je důležitější, aby se data měnila, než aby se na ně aplikovaly jednoduché pravidla.

Doufám, že to pomůže!  Pokud máš další otázky, klidně se ptej.
```

### V2
- Stav: `completed`.
- Delka odpovedi: `307` slov, `2084` znaku.
- Cas generovani: `12.463` s.
- Subjektivni kvalita: pouzitelna - obsahuje hlavni pojmy, ale je strucnejsi.
- Opakovani: opak.
- Kreativita: stredni.
- Konzistence: stredni.

```text
Představ si, že se učíš, jak rozpoznávat kočky a psy. Oba se snaží rozpoznat věci, ale používají trochu jiný přístup. CNN a FFNN (feedforward neural networks) jsou oba typy neuronových sítí, ale mají zásadní rozdíly. Zkus to takhle:

**1. FFNN (Feedforward Neural Network) - Jako jednoduchý "vnímavý systém"**

*   **Jak to funguje:** Představ si, že máš jednoduchý "vnímavý systém" - např. jednoduchý citový systém, který na základě několika znaků (např. barvy, tvaru) rozhodne, jestli je to kočka nebo pes. FFNN funguje podobně. Data (např. obrázek) prochází sítí, a každé "značky" (např. barva) se "zobrazuje" na "vnímavý systém".  Soudí, zda je to kočka nebo pes.
*   **Jak se učí:**  Učí se, jak se data vztahují k "značkám".  Je to jako jednoduchá "přeskačka".  Nemá žádné "přepínací" kroky, které se vnitřně "přeskakují" mezi různými vrstvami.
*   **Jednoduchost:** Velmi jednoduché a rychlé.  Ideální pro jednoduché problémy, kde je to možné.

**2. CNN (Convolutional Neural Network) - Jako "skupina vizualizátorů"**

*   **Jak to funguje:** Představ si, že máš fotku. CNN se soustředí na *detaily* v fotce, jako jsou linie, hrany, barvy.  To dělá pomocí **filtrov** (jakékoliv "přepínací" kroky) -  filtrov, které se "skupí" na různých částech obrazu.  
*   **Jak se učí:**  Učí se, jak rozpoznat *objekty* v obrázku.  Snaží se najít *značky* (např. barvy, textury) v obrázku a naučit se, jak je kombinovat.  Je to jako "skupina vizualizátorů", které se učí *soustředěně* na různých aspektech obrázku.
*   **Výhody:**  Velmi efektivní pro rozpoznávání obrázků a videa, protože dokáže detekovat *víc* detaily na obrázku najednou.  Je to jako "skupina vizualizátorů" které se učí *soustředěně* na různých aspektech obrázku.

**Zjednodušeně:**

*   **FFNN:**  Jednoduchá "přeskačka".
*   **CNN:**  "Skupina vizualizátorů" pro "detaily".

**Příklad:**

*   **FFNN:**  Rozpoznávání jednoduchých tvarů, jako je "kula" nebo "triló".
*   **CNN:**  Rozpoznávání fotek zvířat, jako je kočka, pes, pták.

Doufám, že ti to pomůže pochopit rozdíl!  Pokud máš další otázky, klidně se ptej.
```

### V3
- Stav: `completed`.
- Delka odpovedi: `338` slov, `2131` znaku.
- Cas generovani: `12.603` s.
- Subjektivni kvalita: pouzitelna - obsahuje hlavni pojmy, ale je strucnejsi.
- Opakovani: opak.
- Kreativita: stredni.
- Konzistence: stredni.

```text
Představ si, že se seznámíš s obrázky a naučíš se je rozpoznávat. CNN (Convolutional Neural Network) a FFNN (Feedforward Neural Network) jsou jakési "specializované" programy, které se učí zobecňovat z obrázků. Pojďme si to vysvětlit jednoduše:

**FFNN (Feedforward Neural Network) - Jako zahradník, který se postará o vše**

* **Jak to funguje:** Představ si, že se postará o všechno – zasadí, zalév, kontroluje škůdce. FFNN má malý vnitřní "čtecí pásek" (články a vlnovesnice), které se dívají na obrázek a "vytvoří vzory".  Tyto vzory se pak "učí" rozlišovat mezi různými objekty na obrázku.
* **Jak se učí:** Učí se na základě proměnných a "přinutí" si vzory, které vidí na obrázku. Je to jako naučit se, že červená barva znamená "kočka", zelená znamená "dstání", atd.
* **Příklad použití:** Pokud máš obrázek katů, FFNN se naučí rozpoznávat, že na obrázku jsou katy.
* **Zápas:** Efektivní, ale obtížné se učit z komplexních obrazů.

**CNN (Convolutional Neural Network) - Jako profesionál, který upřednostňuje prostor**

* **Jak to funguje:**  CNN pracují s obrázky jako s "vhodně provedeným terénem". Věnují se specifickým "meziměrovým okromům" na obrázku. Představ si, že se soustředí na určité části obrázku, a soustředí se na *nejdůležitější* věci.
* **Jak se učí:** Vytvoří "pěkné 'mezerky' nebo 'šířky' (konvoluce) a vnímá, kde jsou tyto mezery, a to se často používá k detekci objektů na obrázku – tohle je to, co si "uvědomí".
* **Příklad použití:** Pokud máš fotku krásně způsobeného byاشة, CNN prohledá fotku do místa, kde je uvidíme nejvíce, bez zbytečného zbytečného, a to velmi rychle.
* **Zápas:** Velmi dobré na obrázky s výškovým vzorem: plese, fotografické portréty atd.

**Zjednodušeně:**

* **FFNN** rozpoznává obecné vzory.
* **CNN** se zaměřuje na vzory v obrázcích - obzvlášť kde jsou "mezi" a složitost.

**Praktický příklad:**

* Będzte se učit, že se na snímku vyskytuje kočka. FFNN by se začal do určitého bodu rozprostří a zkoumal, a to bez rozdělení na jednotlivé částí.
* CNN by se musel soustředit na vyvažované oblasti obrázku a dělat tak to, co je pro to pravděpodobné.

Doufám, že to pomůže!

```

### V4
- Stav: `completed`.
- Delka odpovedi: `303` slov, `2182` znaku.
- Cas generovani: `11.516` s.
- Subjektivni kvalita: pouzitelna - obsahuje hlavni pojmy, ale je strucnejsi.
- Opakovani: opak.
- Kreativita: stredni.
- Konzistence: stredni.

```text
Představ si, že se učíš rozpoznávat kočky a psy. Oba typy neuronových sítí, CNN a FFNN, se snaží najít vzory v datech, ale fungují trochu jinak. Zde je rozdíl v jednoduchém jazyce:

**FFNN (Feedforward Neural Network) - Jako jednoduchý "vstupní" systém:**

* **Jak to funguje:** Představ si, že máš vstupní data (např. obrázek kočky) a je je zpracovává.  Sítě FFNN zpracovávají data postupně, jako by se jednalo o "přepraving" dat.  Každá vrstva zpracovává data a přenáší informace na další vrstvu.
* **Představ si to jako:** Jednoduchý "vstupní" systém, který se snaží najít vzory v datech.  Je to jako jednoduchá "přeprava" dat.
* **Použití:**  Dobré pro jednoduché problémy, kde je jasné, co se má zpracovávat.  Například, rozpoznat jednoduché obrázky.

**CNN (Convolutional Neural Network) - Jako "vytvářič" s "vytvářovacími filtry":**

* **Jak to funguje:** CNN se zaměřuje na **vztahy mezi pixely** v obrázku.  Místo toho, aby se snažily najít vzory, CNN používá "vytvářovací filtry" (jakékoliv malé "vytvářecí" operace) k "vytvoření" lokálních vzorů.
* **Představ si to jako:**  Vytvoření "vytvářovacího filtru" na každém pixlu.  Tento filtr se zaměřuje na specifické vzory (např. okraje, barvy) v obrázku.  Pak se tyto vzory "zaznamenávají" a CNN používá tyto vzory k rozpoznání objektu.
* **Použití:**  Velmi dobré pro rozpoznávání obrázků, jako jsou kočky, psi, auta, nebo i obrázky s detaily.  CNN se specializují na zpracování obrazů.

**Zjednodušeně:**

* **FFNN:**  Zpracovává data postupně, jako by se jednalo o "přepravedání".
* **CNN:**  Používá "vytvářecí filtry" k "vytvoření" vzorů a následně je používá k rozpoznání objektu.

**Klíčové rozdíly v krátkém úseku:**

| Funkce          | FFNN                       | CNN                         |
|------------------|-----------------------------|------------------------------|
| **Zpracování**    | Postupně                   | "Vytvářič" s "vytvářovacími filtry" |
| **Vztahy**       | Neřeší vzory              | Zaměřuje se na vzory          |
| **Použití**       | Jednoduché problémy          | Obrázkové problémy (např. rozpoznávání objektů) |

Doufám, že to pomůže!  Pokud máš další otázky, klidně se ptej.
```

### V5
- Stav: `completed`.
- Delka odpovedi: `340` slov, `2364` znaku.
- Cas generovani: `14.162` s.
- Subjektivni kvalita: pouzitelna - obsahuje hlavni pojmy, ale je strucnejsi.
- Opakovani: opak.
- Kreativita: stredni.
- Konzistence: stredni.

```text
Představ si, že se učíš rozpoznávat kočky a psi. Oba systémy - CNN a FFNN (Feedforward Neural Network) - se snaží najít vzory v datech, ale fungují trochu jinak.

**FFNN (Feedforward Neural Network) - jako ten jednoduchý "přeskočovač"**

*   **Jak to funguje:** Představ si, že máš jeden "přeskočovač", který přijímá data, zpracovává je a dává výsledek.  Všechno prochází jen jedním směrem.  Základní principy jsou:
    *   **Vstup:** Data, které se dostanou do systému.
    *   **Zpracování:**  Systém se snaží najít nějaké vzory v datech.
    *   **Výstup:**  Výsledek, který se získá na základě toho, co se seřídil.
*   **Jednoduchost:** Velmi jednoduchý, protože se soustředí na prostý výpočet.
*   **Příklad:**  Představ si, že máš víc než jedno "značky" na obrázku, které se vrací "ano" nebo "ne". FFNN by se snažil najít vzor, který je v těchto značkách.

**CNN (Convolutional Neural Network) - jako "odborník"**

*   **Jak to funguje:**  CNN se specializuje na **viditelné vzory** v datech, jako jsou obrázky.  Místo toho, aby se soustředil na všechny detaily, se soustředí na **menší oblasti** dat.
    *   **Zpracování (Convolution):**  Používá "matice" (vytvořenou z obrázků), která "zkouší" různé části obrázku.  Zálibně hledá **vztahy** mezi různými částmi obrázku.
    *   **Zpracování (Pooling):**  Zmenšuje velké oblasti obrázku, aby se snížila velikost dat a zvýšila robustnost.
*   **Využití:**  CNN jsou skvělé pro rozpoznávání obrazů a videa, ale můžou být použity i pro jiné typy dat, jako je audio nebo text.
*   **Příklad:** Představ si, že máš obrázek fotky kočky. CNN se soustředí na malé oblasti obrázku (např. kříže a čáry) a naučí se, že kočka má obvykle kříže a čáry.

**Shrnutí v tabulce:**

| Funkce           | FFNN                         | CNN                             |
| ----------------- | ---------------------------- | -------------------------------- |
| **Zaměření**       |  Všechny detaily              | Vztahy mezi oblastmi dat      |
| **Velikost**        | Velmi velké, jednodušné     | Menší, složitější              |
| **Využití**        | Obecná analýza dat           | Zpracování obrazů, videa, audio |

**Důležité:** CNN jsou obecně lepší pro složité obrazy a videa, protože se dokážou učit složité vzory. FFNN jsou jednodušší, ale méně efektivní pro složité data.

Doufám, že ti to pomohlo pochopit rozdíl!

```

## 8. Porovnani vystupu
- `V1`: Konzervativni nastaveni, vhodne pro stabilni a vecne vysvetleni.
- `V2`: Vyrovnane nastaveni mezi stabilitou a pestrostí formulace.
- `V3`: Kreativnejsi nastaveni, muze byt rozmanitejsi, ale s vyssim rizikem mene presne formulace.
- `V4`: Konzervativni nastaveni, vhodne pro stabilni a vecne vysvetleni.
- `V5`: Vetsi kontext je pripraveny pro delsi zadani, u tohoto kratkeho promptu nema zasadni vyhodu.

## 9. Diskuze
Temperature zvysuje nahodnost generovani. Nizka hodnota je vhodna pro konzistentni vyukove odpovedi, vyssi hodnota muze prinest pestrejsi formulace, ale i vice odchylek.
`top_p` omezuje vyber tokenu podle kumulativni pravdepodobnosti. Nizsi hodnota obvykle zpusobi soustredenejsi odpoved, vyssi hodnota ponecha modelu vice moznosti.
`top_k` omezuje pocet kandidatu pro dalsi token. Mensi hodnota podporuje konzervativnejsi text, vetsi hodnota muze rozsirit slovnik a styl.
`repeat_penalty` tresta opakovani. Vyssi hodnota je uzitecna, pokud model zacina opakovat stejne obraty, ale prilis vysoka muze narusit prirozenost textu.
`num_ctx` urcuje velikost kontextoveho okna. U kratkeho promptu se rozdil nemusi projevit, ale pro dlouhe vstupy nebo vice prikladu je vyssi kontext uzitecny.
Konzervativni nastaveni se hodi pro fakticke vysvetleni a vyuku. Kreativnejsi nastaveni se hodi pro brainstorming, varianty formulaci nebo priklady.

## 10. Zaver
Experiment je pripraveny tak, aby pri dostupne Ollame ulozil realne odpovedi do JSON a souhrn do CSV. Pokud Ollama nebezi, skript skonci s jasnym fallbackem a nevymysli odpovedi modelu.
