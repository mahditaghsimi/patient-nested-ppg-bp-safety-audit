#!/usr/bin/env python3
"""Build the published MIMIC-BP 1100/195/229 split with explicit duplicate policy."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/mimic_bp/splits/official_source"
PPG = ROOT / "data/mimic_bp/raw/ppg"
OUT = ROOT / "data/mimic_bp/splits/official_patient_split.json"


def read_list(name: str) -> list[str]:
    value = ast.literal_eval((SOURCE / name).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Invalid official split file: {name}")
    return value


def main() -> None:
    split = {
        "train": read_list("train_subjects.txt"),
        "validation": read_list("val_subjects.txt"),
        "test": read_list("test_subjects.txt"),
    }
    expected = {"train": 1100, "validation": 195, "test": 229}
    if {key: len(value) for key, value in split.items()} != expected:
        raise RuntimeError("Official MIMIC-BP split counts do not match 1100/195/229")
    sets = {key: set(value) for key, value in split.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Official split contains patient overlap")
    local = {path.name.removesuffix("_ppg.npy") for path in PPG.glob("p*_ppg.npy")}
    if set.union(*sets.values()) != local:
        raise RuntimeError("Official split IDs do not exactly match the local 1,524 PPG patients")

    hashes: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(PPG.glob("p*_ppg.npy")):
        patient = path.name.removesuffix("_ppg.npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        for segment, row in enumerate(array):
            digest = hashlib.blake2b(memoryview(np.ascontiguousarray(row)), digest_size=16).hexdigest()
            hashes[digest].append({"patient_id": patient, "segment_id": segment})
    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    excluded: set[str] = set()
    policy = []
    for group in duplicate_groups:
        patients = {item["patient_id"] for item in group}
        removed = group if len(patients) > 1 else group[1:]
        excluded.update(f"{item['patient_id']}:{item['segment_id']}" for item in removed)
        policy.append({"group": group, "removed": removed, "reason": "all copies removed across patients" if len(patients) > 1 else "retain first within-patient copy"})

    payload = {
        "source": "published MIMIC-BP subject lists bundled with the dataset",
        "policy": "official 1100/195/229 patient-disjoint split; exact PPG duplicates removed before modeling",
        "seed": None,
        **split,
        "excluded_segments": sorted(excluded),
        "duplicate_policy": policy,
        "counts": {
            "train_patients": 1100, "validation_patients": 195, "test_patients": 229,
            "raw_segments": 45720, "excluded_segments": len(excluded), "retained_segments": 45720 - len(excluded),
            "duplicate_groups": len(duplicate_groups),
            "cross_patient_duplicate_groups": sum(len({item['patient_id'] for item in group}) > 1 for group in duplicate_groups),
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(SOURCE.glob("*_subjects.txt"))
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
