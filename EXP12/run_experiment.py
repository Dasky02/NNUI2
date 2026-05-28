from __future__ import annotations

import csv
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
TAGS_ENDPOINT = "http://localhost:11434/api/tags"
MODEL = "gemma3:1b"
PROMPT = "Vysvětli rozdíl mezi CNN a FFNN jednoduše pro studenta."

VARIANTS: list[dict[str, Any]] = [
    {"variant": "V1", "temperature": 0.2, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_ctx": 2048},
    {"variant": "V2", "temperature": 0.8, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_ctx": 2048},
    {"variant": "V3", "temperature": 1.2, "top_p": 1.0, "top_k": 100, "repeat_penalty": 1.05, "num_ctx": 2048},
    {"variant": "V4", "temperature": 0.2, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.2, "num_ctx": 2048},
    {"variant": "V5", "temperature": 0.7, "top_p": 0.95, "top_k": 80, "repeat_penalty": 1.1, "num_ctx": 8192},
]


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_version() -> str:
    try:
        completed = subprocess.run(["ollama", "--version"], check=False, text=True, capture_output=True, timeout=5)
    except Exception as exc:
        return f"unavailable: {exc!r}"
    return (completed.stdout or completed.stderr).strip() or "unknown"


def available_models() -> list[str]:
    try:
        payload = request_json(TAGS_ENDPOINT, timeout=3)
    except Exception:
        return []
    return [str(model.get("name")) for model in payload.get("models", [])]


def environment_info(model: str) -> dict[str, Any]:
    mem_bytes = None
    try:
        completed = subprocess.run(["sysctl", "-n", "hw.memsize"], check=False, text=True, capture_output=True, timeout=3)
        mem_bytes = int((completed.stdout or "0").strip()) or None
    except Exception:
        mem_bytes = None
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "ollama_version": ollama_version(),
        "model": model,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(mem_bytes / (1024**3), 2) if mem_bytes else None,
        "endpoint": DEFAULT_ENDPOINT,
    }


def call_ollama(model: str, variant: dict[str, Any], prompt: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": variant["temperature"],
            "top_p": variant["top_p"],
            "top_k": variant["top_k"],
            "repeat_penalty": variant["repeat_penalty"],
            "num_ctx": variant["num_ctx"],
        },
    }
    started = time.perf_counter()
    response = request_json(DEFAULT_ENDPOINT, payload=payload, timeout=180)
    wall_time = time.perf_counter() - started
    text = str(response.get("response", ""))
    total_duration = response.get("total_duration")
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    return {
        "variant": variant["variant"],
        "parameters": {key: variant[key] for key in ["temperature", "top_p", "top_k", "repeat_penalty", "num_ctx"]},
        "status": "completed",
        "response": text,
        "response_length_chars": len(text),
        "response_length_words": len(text.split()),
        "wall_time_s": round(wall_time, 3),
        "ollama_total_duration_s": round(float(total_duration) / 1_000_000_000, 3) if total_duration else "",
        "eval_count": eval_count if eval_count is not None else "",
        "tokens_per_second": round(float(eval_count) / (float(eval_duration) / 1_000_000_000), 2)
        if eval_count and eval_duration
        else "",
        "raw": response,
    }


def quality_notes(text: str, variant: dict[str, Any]) -> dict[str, str]:
    lower = text.lower()
    keyword_hits = sum(1 for word in ["cnn", "ffnn", "konvol", "obraz", "vrstv", "student"] if word in lower)
    repeated = "opak" if len(text.split()) != len(set(text.lower().split())) else "bez zjevneho opakovani"
    if not text:
        quality = "bez odpovedi"
        creativity = "nelze hodnotit"
        consistency = "nelze hodnotit"
    elif keyword_hits >= 5 and len(text.split()) >= 50:
        quality = "dobra - pokryva rozdil CNN/FFNN a je dostatecne konkretni"
        consistency = "vysoka" if float(variant["temperature"]) <= 0.8 else "stredni"
        creativity = "nizka az stredni" if float(variant["temperature"]) <= 0.7 else "vyssi"
    elif keyword_hits >= 3:
        quality = "pouzitelna - obsahuje hlavni pojmy, ale je strucnejsi"
        consistency = "stredni"
        creativity = "stredni"
    else:
        quality = "slabsi - chybi cast klicovych pojmu"
        consistency = "nizka"
        creativity = "nevyhodnoceno"
    return {
        "subjective_quality": quality,
        "repetition": repeated,
        "creativity": creativity,
        "consistency": consistency,
    }


def write_summary_csv(outputs: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "status",
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "num_ctx",
        "response_length_chars",
        "response_length_words",
        "wall_time_s",
        "ollama_total_duration_s",
        "eval_count",
        "tokens_per_second",
        "subjective_quality",
        "repetition",
        "creativity",
        "consistency",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for output in outputs:
            params = output.get("parameters", {})
            notes = output.get("quality_notes", {})
            writer.writerow(
                {
                    "variant": output.get("variant"),
                    "status": output.get("status"),
                    "temperature": params.get("temperature", ""),
                    "top_p": params.get("top_p", ""),
                    "top_k": params.get("top_k", ""),
                    "repeat_penalty": params.get("repeat_penalty", ""),
                    "num_ctx": params.get("num_ctx", ""),
                    "response_length_chars": output.get("response_length_chars", ""),
                    "response_length_words": output.get("response_length_words", ""),
                    "wall_time_s": output.get("wall_time_s", ""),
                    "ollama_total_duration_s": output.get("ollama_total_duration_s", ""),
                    "eval_count": output.get("eval_count", ""),
                    "tokens_per_second": output.get("tokens_per_second", ""),
                    "subjective_quality": notes.get("subjective_quality", ""),
                    "repetition": notes.get("repetition", ""),
                    "creativity": notes.get("creativity", ""),
                    "consistency": notes.get("consistency", ""),
                    "error": output.get("error", ""),
                }
            )


def variant_table() -> list[str]:
    lines = [
        "| Varianta | temperature | top_p | top_k | repeat_penalty | num_ctx |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        lines.append(
            f"| `{variant['variant']}` | {variant['temperature']} | {variant['top_p']} | "
            f"{variant['top_k']} | {variant['repeat_penalty']} | {variant['num_ctx']} |"
        )
    return lines


def result_sections(outputs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for output in outputs:
        notes = output.get("quality_notes", {})
        lines.extend(
            [
                f"### {output['variant']}",
                f"- Stav: `{output.get('status')}`.",
                f"- Delka odpovedi: `{output.get('response_length_words', '-')}` slov, `{output.get('response_length_chars', '-')}` znaku.",
                f"- Cas generovani: `{output.get('wall_time_s', '-')}` s.",
                f"- Subjektivni kvalita: {notes.get('subjective_quality', output.get('error', 'nevyhodnoceno'))}.",
                f"- Opakovani: {notes.get('repetition', '-')}.",
                f"- Kreativita: {notes.get('creativity', '-')}.",
                f"- Konzistence: {notes.get('consistency', '-')}.",
                "",
                "```text",
                output.get("response", ""),
                "```",
                "",
            ]
        )
    return lines


def comparison_sections(outputs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for output in outputs:
        params = output.get("parameters", {})
        temp = params.get("temperature")
        ctx = params.get("num_ctx")
        if output.get("status") != "completed":
            comment = "Varianta nebyla vygenerovana, proto ji nelze kvalitativne porovnat."
        elif temp is not None and float(temp) <= 0.2:
            comment = "Konzervativni nastaveni, vhodne pro stabilni a vecne vysvetleni."
        elif temp is not None and float(temp) >= 1.0:
            comment = "Kreativnejsi nastaveni, muze byt rozmanitejsi, ale s vyssim rizikem mene presne formulace."
        elif ctx == 8192:
            comment = "Vetsi kontext je pripraveny pro delsi zadani, u tohoto kratkeho promptu nema zasadni vyhodu."
        else:
            comment = "Vyrovnane nastaveni mezi stabilitou a pestrostí formulace."
        lines.append(f"- `{output['variant']}`: {comment}")
    return lines


def write_report(report_path: Path, outputs: list[dict[str, Any]], env: dict[str, Any], status: str, error: str | None) -> None:
    lines = [
        "# NNUI2 - Cviceni 12: Lokalni LLM pomoci Ollama",
        "",
        "## 1. Cil",
        "Cilem je overit lokalni spusteni LLM pres Ollama, zavolat model pres HTTP API a porovnat vliv parametru generovani na odpoved na jeden pevny prompt.",
        "",
        "## 2. Prostredi",
        f"- OS: `{env['os']}`.",
        f"- Python: `{env['python']}`.",
        f"- Ollama: `{env['ollama_version']}`.",
        f"- Pouzity model: `{env['model']}`.",
        f"- CPU/GPU/RAM: CPU `{env['cpu']}`, CPU count `{env['cpu_count']}`, RAM `{env['ram_gb']} GB`; GPU nebylo pres skript samostatne zjistovano.",
        f"- Dostupnost API: `{status}`" + (f" (`{error}`)." if error else "."),
        "",
        "## 3. Spusteni Ollama",
        "```bash",
        "ollama serve",
        f"ollama pull {env['model']}",
        "```",
        "",
        "## 4. API volani",
        f"- Endpoint: `{DEFAULT_ENDPOINT}`.",
        "- Python request pouzity ve skriptu:",
        "```python",
        "payload = {",
        f"    \"model\": \"{env['model']}\",",
        "    \"prompt\": PROMPT,",
        "    \"stream\": False,",
        "    \"options\": {\"temperature\": 0.2, \"top_p\": 0.9, \"top_k\": 40, \"repeat_penalty\": 1.1, \"num_ctx\": 2048},",
        "}",
        "```",
        "",
        "## 5. Testovany prompt",
        f"`{PROMPT}`",
        "",
        "## 6. Testovane varianty parametru",
        *variant_table(),
        "",
        "## 7. Vysledky",
        *result_sections(outputs),
        "## 8. Porovnani vystupu",
        *comparison_sections(outputs),
        "",
        "## 9. Diskuze",
        "Temperature zvysuje nahodnost generovani. Nizka hodnota je vhodna pro konzistentni vyukove odpovedi, vyssi hodnota muze prinest pestrejsi formulace, ale i vice odchylek.",
        "`top_p` omezuje vyber tokenu podle kumulativni pravdepodobnosti. Nizsi hodnota obvykle zpusobi soustredenejsi odpoved, vyssi hodnota ponecha modelu vice moznosti.",
        "`top_k` omezuje pocet kandidatu pro dalsi token. Mensi hodnota podporuje konzervativnejsi text, vetsi hodnota muze rozsirit slovnik a styl.",
        "`repeat_penalty` tresta opakovani. Vyssi hodnota je uzitecna, pokud model zacina opakovat stejne obraty, ale prilis vysoka muze narusit prirozenost textu.",
        "`num_ctx` urcuje velikost kontextoveho okna. U kratkeho promptu se rozdil nemusi projevit, ale pro dlouhe vstupy nebo vice prikladu je vyssi kontext uzitecny.",
        "Konzervativni nastaveni se hodi pro fakticke vysvetleni a vyuku. Kreativnejsi nastaveni se hodi pro brainstorming, varianty formulaci nebo priklady.",
        "",
        "## 10. Zaver",
        "Experiment je pripraveny tak, aby pri dostupne Ollame ulozil realne odpovedi do JSON a souhrn do CSV. Pokud Ollama nebezi, skript skonci s jasnym fallbackem a nevymysli odpovedi modelu.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = Path(__file__).resolve().parent
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)
    models = available_models()
    model = MODEL if MODEL in models else (models[0] if models else MODEL)
    env = environment_info(model)
    outputs: list[dict[str, Any]] = []
    status = "available"
    error: str | None = None

    if not models:
        status = "unavailable"
        error = "Ollama API is not reachable or no local models are available"
        for variant in VARIANTS:
            outputs.append(
                {
                    "variant": variant["variant"],
                    "parameters": {key: variant[key] for key in ["temperature", "top_p", "top_k", "repeat_penalty", "num_ctx"]},
                    "status": "blocked",
                    "response": "",
                    "error": error,
                }
            )
    else:
        for variant in VARIANTS:
            try:
                output = call_ollama(model, variant, PROMPT)
                output["quality_notes"] = quality_notes(output["response"], variant)
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                status = "partial"
                output = {
                    "variant": variant["variant"],
                    "parameters": {key: variant[key] for key in ["temperature", "top_p", "top_k", "repeat_penalty", "num_ctx"]},
                    "status": "failed",
                    "response": "",
                    "error": f"Ollama request failed: {exc!r}",
                }
                output["quality_notes"] = quality_notes("", variant)
            outputs.append(output)

    payload = {
        "status": status,
        "error": error,
        "environment": env,
        "prompt": PROMPT,
        "outputs": outputs,
    }
    (results / "ollama_outputs.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(outputs, results / "summary.csv")
    write_report(ROOT / "experiment_12.md", outputs, env, status, error)
    write_report(base / "report.md", outputs, env, status, error)

    print("EXP12 completed")
    print(f"Ollama status: {status}")
    print(f"Model: {model}")
    print(f"Outputs: {results / 'ollama_outputs.json'}")
    print(f"Report: {ROOT / 'experiment_12.md'}")
    return 0 if status in {"available", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
