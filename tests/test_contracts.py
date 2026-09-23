from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_official_patient_split_contract():
    split = json.loads((ROOT / "data/mimic_bp/splits/official_patient_split.json").read_text())
    groups = {name: set(split[name]) for name in ("train", "validation", "test")}
    assert {name: len(values) for name, values in groups.items()} == {
        "train": 1100,
        "validation": 195,
        "test": 229,
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert len(set.union(*groups.values())) == 1524
    assert split["counts"]["retained_segments"] == 45689
    assert len(split["excluded_segments"]) == 31


def test_innovation_registry_is_exactly_200_and_unique():
    from phase2.innovation200_registry import all_configs

    configs = all_configs()
    assert len(configs) == 200
    assert len({row["run_id"] for row in configs}) == 200
    stage_counts = {}
    for row in configs:
        stage_counts[row["stage"]] = stage_counts.get(row["stage"], 0) + 1
    assert stage_counts == {"ARCH": 80, "INPUT": 40, "OBJECTIVE": 40, "TRAIN": 40}


def test_internal_cache_contract():
    metadata = np.load(ROOT / "outputs/neural/cache/metadata.npz", allow_pickle=True)
    waveform = np.load(ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy", mmap_mode="r")
    assert waveform.shape == (45689, 3, 1875)
    assert metadata["y"].shape == (45689, 2)
    assert len(np.unique(metadata["patient_id"])) == 1524
    assert set(np.unique(metadata["split"])) == {"train", "validation", "test"}
    assert {
        part: len(np.unique(metadata["patient_id"][metadata["split"] == part]))
        for part in ("train", "validation", "test")
    } == {"train": 1100, "validation": 195, "test": 229}
    manifest = json.loads((ROOT / "outputs/neural/cache/manifest.json").read_text())
    assert manifest["source_window"].startswith("central 15 s")
    feature_manifest = json.loads((ROOT / "outputs/features/FEATURE_CACHE_MANIFEST.json").read_text())
    assert feature_manifest["source_window"].startswith("central 15 s")


def test_external_contracts():
    contracts = {
        "data/external/prepared/ppg_bp_219.npz": ((657, 3, 1875), 219),
        "data/external/prepared/ppg_ambulatory_56.npz": ((461, 3, 1875), 56),
        "data/external/vitaldb/prepared/vitaldb_sync128.npz": ((2380, 3, 1875), 119),
    }
    for relative, (shape, patients) in contracts.items():
        values = np.load(ROOT / relative, allow_pickle=True)
        assert values["x"].shape == shape
        assert len(np.unique(values["patient_id"])) == patients


def test_primary_decision_contract():
    decision = json.loads((ROOT / "phase2/outputs/innovation200_confirmation/DECISION.json").read_text())
    assert decision["screening_methods"] == 200
    assert decision["confirmation_fits"] == 75
    assert decision["status"] in {"NO_CONFIRMED_IMPROVEMENT", "CONFIRMED_IMPROVEMENT"}
    assert isinstance(decision["winner_id"], str) and decision["winner_id"]


def test_nested_selection_contract():
    decision = json.loads((ROOT / "phase2/outputs/nested200/DECISION.json").read_text())
    assert decision["status"] == "PASS"
    assert decision["new_methods"] == 200
    assert decision["methods_per_inner_screen"] == 201
    assert decision["inner_fits"] == 1005
    assert decision["outer_patient_overlap"] == 0
    assert decision["inner_outer_test_overlap"] == 0
    assert decision["each_segment_predicted_once"] is True
