#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from common import load_config

    report: dict[str, object] = {"checked_utc": datetime.now(timezone.utc).isoformat(), "checks": []}
    dependencies = {}
    for module in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib", "yaml", "docx", "openpyxl"):
        loaded = importlib.import_module(module)
        dependencies[module] = getattr(loaded, "__version__", "installed")
    report["checks"].append({"name": "dependencies", "status": "PASS", "versions": dependencies})

    config = load_config()
    for key in ("data_dir", "labels_dir", "split_path", "forecast_checkpoint"):
        check(Path(config[key]).exists(), f"Missing configured path: {key}={config[key]}")

    ppg_files = sorted(Path(config["data_dir"]).glob("p*_ppg.npy"))
    label_files = sorted(Path(config["labels_dir"]).glob("p*_labels.npy"))
    check(len(ppg_files) == 1524, f"Expected 1524 PPG files, found {len(ppg_files)}")
    check(len(label_files) == 1524, f"Expected 1524 label files, found {len(label_files)}")
    check(np.load(ppg_files[0], mmap_mode="r").shape == (30, 3750), "Unexpected raw PPG shape")
    check(np.load(label_files[0], mmap_mode="r").shape == (30, 2), "Unexpected raw label shape")
    report["checks"].append({"name": "internal_raw", "status": "PASS", "patients": 1524, "segments_per_patient": 30})

    metadata_path = ROOT / "outputs/neural/cache/metadata.npz"
    signal_path = ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy"
    feature_path = ROOT / "outputs/features/feature_cache.npz"
    for path in (metadata_path, signal_path, feature_path):
        check(path.exists(), f"Missing derived cache: {path}")
    metadata = np.load(metadata_path, allow_pickle=True)
    check(metadata["y"].shape == (45689, 2), "Expected 45,689 deduplicated labels")
    split_counts = {str(value): int(np.sum(metadata["split"] == value)) for value in np.unique(metadata["split"])}
    check(split_counts == {"test": 6868, "train": 32971, "validation": 5850}, f"Official split mismatch: {split_counts}")
    patient_counts = {
        name: int(np.unique(metadata["patient_id"][metadata["split"].astype(str) == name]).size)
        for name in ("train", "validation", "test")
    }
    check(patient_counts == {"train": 1100, "validation": 195, "test": 229}, f"Official patient split mismatch: {patient_counts}")
    waveform = np.load(signal_path, mmap_mode="r")
    check(waveform.shape == (45689, 3, 1875), f"Waveform cache mismatch: {waveform.shape}")
    report["checks"].append({"name": "internal_derived", "status": "PASS", "shape": list(waveform.shape), "split_segments": split_counts, "split_patients": patient_counts})

    external_contracts = {
        "ppg_bp": (ROOT / "data/external/prepared/ppg_bp_219.npz", (657, 3, 1875), 219),
        "ambulatory": (ROOT / "data/external/prepared/ppg_ambulatory_56.npz", (461, 3, 1875), 56),
        "vitaldb": (ROOT / "data/external/vitaldb/prepared/vitaldb_sync128.npz", (2380, 3, 1875), 119),
    }
    external = {}
    for name, (path, shape, patients) in external_contracts.items():
        check(path.exists(), f"Missing external prepared dataset: {path}")
        values = np.load(path, allow_pickle=True)
        check(values["x"].shape == shape, f"{name} shape mismatch: {values['x'].shape}")
        found_patients = int(np.unique(values["patient_id"]).size)
        check(found_patients == patients, f"{name} patient count mismatch: {found_patients}")
        external[name] = {"shape": list(shape), "patients": patients}
    report["checks"].append({"name": "external_data", "status": "PASS", "datasets": external})

    weight = ROOT / "data/pretraining/papagei_s.pt"
    check(weight.exists() and weight.stat().st_size > 1_000_000, "Missing PaPaGei-S checkpoint")
    large = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size > 100_000_000)
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    check("*.npy filter=lfs" in attributes and "*.npz filter=lfs" in attributes, "Git LFS rules missing")
    report["checks"].append({"name": "github_large_files", "status": "PASS", "requires_git_lfs": large})

    decision = ROOT / "phase2/outputs/innovation200_confirmation/DECISION.json"
    check(decision.exists(), "Missing primary decision artifact")
    values = json.loads(decision.read_text(encoding="utf-8"))
    check(values["screening_methods"] == 200 and values["confirmation_fits"] == 75, "Incomplete primary experiment")
    check(values["status"] in {"NO_CONFIRMED_IMPROVEMENT", "CONFIRMED_IMPROVEMENT"}, "Unknown primary decision status")
    report["checks"].append({"name": "reference_results", "status": "PASS", "decision": values["status"]})

    report["status"] = "PASS"
    destination = ROOT / "artifacts/validation_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
