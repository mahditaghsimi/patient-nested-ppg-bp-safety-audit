#!/usr/bin/env python3
"""Fail early with a precise list of missing user-supplied inputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.is_dir() else 0


def main() -> None:
    checks = []

    def add(name: str, passed: bool, evidence: object, remedy: str) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence, "remedy": remedy})

    ppg = ROOT / "data/mimic_bp/raw/ppg"
    labels = ROOT / "data/mimic_bp/raw/labels"
    ppg_count = count(ppg, "p*_ppg.npy")
    label_count = count(labels, "p*_labels.npy")
    add("mimic_ppg", ppg_count == 1524, {"found": ppg_count, "expected": 1524}, "Place pXXXX_ppg.npy files under data/mimic_bp/raw/ppg/.")
    add("mimic_labels", label_count == 1524, {"found": label_count, "expected": 1524}, "Place pXXXX_labels.npy files under data/mimic_bp/raw/labels/.")

    split_root = ROOT / "data/mimic_bp/splits/official_source"
    split_files = [split_root / name for name in ("train_subjects.txt", "val_subjects.txt", "test_subjects.txt")]
    add("official_split_lists", all(path.is_file() and path.stat().st_size > 0 for path in split_files), [str(path.relative_to(ROOT)) for path in split_files], "Supply the three official MIMIC-BP subject-list files.")
    historical_split = ROOT / "data/mimic_bp/splits/historical_custom_split_seed42.json"
    add("historical_split_audit", historical_split.is_file() and historical_split.stat().st_size > 0, str(historical_split.relative_to(ROOT)), "Supply the historical split JSON used only by the leakage crosswalk audit.")

    forecast = ROOT / "data/pretraining/forecast_encoder_best_model.pt"
    papagei = ROOT / "data/pretraining/papagei_s.pt"
    add("forecast_checkpoint", forecast.is_file() and forecast.stat().st_size > 0, str(forecast.relative_to(ROOT)), "Place the forecasting checkpoint in data/pretraining/.")
    add("papagei_checkpoint", papagei.is_file() and papagei.stat().st_size > 1_000_000, str(papagei.relative_to(ROOT)), "Place the licensed PaPaGei-S checkpoint in data/pretraining/.")

    ppg_bp = ROOT / "data/external/ppg_bp/raw/Data File/PPG-BP dataset.xlsx"
    ppg_bp_signals = ROOT / "data/external/ppg_bp/raw/Data File/0_subject"
    add("ppg_bp", ppg_bp.is_file() and count(ppg_bp_signals, "*.txt") > 0, {"workbook": ppg_bp.is_file(), "signals": count(ppg_bp_signals, "*.txt")}, "Copy the original PPG-BP Data File directory under data/external/ppg_bp/raw/.")

    ambulatory = ROOT / "data/external/ppg_ambulatory_56/raw/PPG_json"
    add("ambulatory_56", count(ambulatory, "*.json") >= 56, {"json_files": count(ambulatory, "*.json")}, "Place the source JSON files under data/external/ppg_ambulatory_56/raw/PPG_json/.")

    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    report = {"status": status, "checks": checks, "vitaldb": "Downloaded by the locked preparation stage or reused from data/external/vitaldb/raw/."}
    destination = ROOT / "artifacts/input_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
