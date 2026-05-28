from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "EXP12"
sys.path.insert(0, str(EXP))

from run_experiment import PROMPT, VARIANTS, quality_notes


def test_five_generation_variants_are_defined():
    assert len(VARIANTS) == 5
    assert [variant["variant"] for variant in VARIANTS] == ["V1", "V2", "V3", "V4", "V5"]
    assert VARIANTS[0]["temperature"] == 0.2
    assert VARIANTS[-1]["num_ctx"] == 8192


def test_prompt_is_fixed_assignment_prompt():
    assert PROMPT == "Vysvětli rozdíl mezi CNN a FFNN jednoduše pro studenta."


def test_quality_notes_handles_empty_response():
    notes = quality_notes("", VARIANTS[0])
    assert notes["subjective_quality"] == "bez odpovedi"
