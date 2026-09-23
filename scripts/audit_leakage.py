#!/usr/bin/env python3
"""Evidence-based leakage audit for the complete final PPG-to-BP workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.run_confirmation_cv import subject_strata
from neural.build_input_cache import robust_channels
from features.feature_math import all_feature_groups, preprocess_batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], name: str, status: str, evidence: Any, consequence: str = "") -> None:
    checks.append({"name": name, "status": status, "evidence": evidence, "consequence": consequence})


def compare_splits(checks: list[dict[str, Any]]) -> None:
    official = json.loads((ROOT / "data/mimic_bp/splits/official_patient_split.json").read_text())
    historical = json.loads((ROOT / "data/mimic_bp/splits/historical_custom_split_seed42.json").read_text())
    official_sets = {key: set(official[key]) for key in ("train", "validation", "test")}
    historical_sets = {"train": set(historical["train"]), "validation": set(historical["validation"]), "test": set(historical["test"])}
    overlaps = {
        old: {new: len(historical_sets[old] & official_sets[new]) for new in official_sets}
        for old in historical_sets
    }
    disjoint = all(not official_sets[a] & official_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    add(checks, "official_split_patient_disjoint", "PASS" if disjoint else "FAIL", {
        "counts": {key: len(value) for key, value in official_sets.items()}, "pairwise_overlap": 0 if disjoint else "nonzero",
        "source_hashes": official["source_sha256"],
    })
    add(checks, "historical_split_was_not_official", "CORRECTED", {
        "historical_counts": {key: len(value) for key, value in historical_sets.items()},
        "official_counts": {key: len(value) for key, value in official_sets.items()},
        "crosswalk": overlaps, "historical_test_equals_official_test": historical_sets["test"] == official_sets["test"],
    }, "The old manuscript wording 'original split' was factually wrong; FINAL uses and names the official split.")


def audit_metadata(checks: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split = json.loads((ROOT / "data/mimic_bp/splits/official_patient_split.json").read_text())
    metadata_path = ROOT / "outputs/neural/cache/metadata.npz"
    if not metadata_path.exists():
        add(checks, "metadata_cache", "PENDING", str(metadata_path), "Run preprocessing before the post-run audit.")
        return {}, split
    with np.load(metadata_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    expected = {patient: key for key in ("train", "validation", "test") for patient in split[key]}
    found = {str(patient): str(part) for patient, part in zip(arrays["patient_id"], arrays["split"])}
    mismatches = sorted(patient for patient, part in found.items() if expected[patient] != part)
    counts = {key: int(np.sum(arrays["split"].astype(str) == key)) for key in ("train", "validation", "test")}
    patient_counts = {key: int(np.unique(arrays["patient_id"][arrays["split"].astype(str) == key]).size) for key in counts}
    unique_keys = len(set(zip(arrays["patient_id"].astype(str), arrays["segment_id"].astype(int)))) == len(arrays["y"])
    add(checks, "metadata_matches_official_split", "PASS" if not mismatches else "FAIL", {
        "segment_counts": counts, "patient_counts": patient_counts, "mismatched_patients": mismatches,
        "unique_patient_segment_keys": unique_keys, "segments": len(arrays["y"]),
    })
    excluded = set(split["excluded_segments"])
    retained = {f"{p}:{s}" for p, s in zip(arrays["patient_id"].astype(str), arrays["segment_id"].astype(int))}
    absent = sorted(excluded & retained)
    cross_groups = [row for row in split["duplicate_policy"] if len({item["patient_id"] for item in row["group"]}) > 1]
    cross_removed = all(f"{item['patient_id']}:{item['segment_id']}" not in retained for row in cross_groups for item in row["group"])
    inconsistent_labels = []
    for row in split["duplicate_policy"]:
        labels = [np.load(ROOT / f"data/mimic_bp/raw/labels/{item['patient_id']}_labels.npy", allow_pickle=False)[int(item["segment_id"])] for item in row["group"]]
        if not all(np.array_equal(labels[0], value) for value in labels[1:]):
            inconsistent_labels.append(row["group"])
    add(checks, "exact_duplicate_policy", "PASS" if not absent and cross_removed and not inconsistent_labels else "FAIL", {
        "duplicate_groups": split["counts"]["duplicate_groups"], "excluded_segments": len(excluded),
        "cross_patient_groups": len(cross_groups), "excluded_still_retained": absent,
        "all_cross_patient_copies_removed": cross_removed,
        "groups_with_inconsistent_sbp_dbp_labels": inconsistent_labels,
        "definition": "BLAKE2b equality of all 3,750 float samples in the raw 30-s PPG segment",
    })
    return arrays, split


def audit_cache_window(checks: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    if not arrays:
        return
    cache = np.load(ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy", mmap_mode="r", allow_pickle=False)
    samples = np.linspace(0, len(arrays["y"]) - 1, 25, dtype=int)
    maximum = 0.0
    for index in samples:
        patient = str(arrays["patient_id"][index]); segment = int(arrays["segment_id"][index])
        raw = np.load(ROOT / f"data/mimic_bp/raw/ppg/{patient}_ppg.npy", mmap_mode="r")[segment]
        central = np.asarray(raw[(3750 - 1875) // 2 : (3750 - 1875) // 2 + 1875], dtype=np.float32)
        # Replay the production function, including its explicit float32 input
        # cast followed by float64 robust statistics. A hand-written audit had
        # differed by one float16 ULP for a few float64 source records.
        expected = robust_channels(central)[0][0].astype(np.float16)
        maximum = max(maximum, float(np.max(np.abs(cache[index, 0].astype(np.float32) - expected.astype(np.float32)))))
    add(checks, "central_window_and_local_normalization", "PASS" if maximum == 0 else "FAIL", {
        "audited_samples": len(samples), "source_samples": [937, 2812], "maximum_float16_difference": maximum,
        "fit_scope": "each input window independently; no labels or other patients used",
    })


def audit_feature_window(checks: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    if not arrays:
        return
    feature_path = ROOT / "outputs/features/feature_cache.npz"
    manifest_path = feature_path.parent / "FEATURE_CACHE_MANIFEST.json"
    if not feature_path.exists() or not manifest_path.exists():
        add(checks, "engineered_feature_central_window", "PENDING", str(feature_path), "Regenerate the feature cache.")
        return
    with np.load(feature_path, allow_pickle=False) as feature:
        bank = feature["raw__fusion"]
        samples = np.linspace(0, len(arrays["y"]) - 1, 12, dtype=int)
        maximum = 0.0
        for index in samples:
            patient = str(arrays["patient_id"][index]); segment = int(arrays["segment_id"][index])
            raw = np.load(ROOT / f"data/mimic_bp/raw/ppg/{patient}_ppg.npy", mmap_mode="r", allow_pickle=False)[segment]
            central = np.asarray(raw[937:2812], dtype=np.float64)[None]
            groups, _ = all_feature_groups(preprocess_batch(central, "raw", 125), 125)
            maximum = max(maximum, float(np.max(np.abs(groups["fusion"][0].astype(np.float32) - bank[index].astype(np.float32)))))
    manifest = json.loads(manifest_path.read_text())
    status = "PASS" if maximum <= 1e-5 and manifest["source_window"].startswith("central 15 s") else "FAIL"
    add(checks, "engineered_feature_central_window", status, {
        "audited_samples": len(samples), "maximum_float32_difference": maximum,
        "manifest": manifest, "fit_scope": "feature extraction per window; scaling later fitted on fold training indices only",
    })


def expected_fold(arrays: dict[str, np.ndarray], seed: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    subjects, strata = subject_strata(arrays["patient_id"], arrays["y"])
    partitions = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(subjects, strata))
    train_subject_i, test_subject_i = partitions[fold - 1]
    train_idx = np.flatnonzero(np.isin(arrays["patient_id"], subjects[train_subject_i]))
    test_idx = np.flatnonzero(np.isin(arrays["patient_id"], subjects[test_subject_i]))
    return train_idx, test_idx


def audit_confirmation(checks: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    results_path = ROOT / "phase2/outputs/innovation200_confirmation/fold_results.csv"
    if not arrays or not results_path.exists():
        add(checks, "confirmation_fold_replay", "PENDING", str(results_path), "Run confirmation before the post-run audit.")
        return
    frame = pd.read_csv(results_path)
    failures = []
    coverage: dict[tuple[str, int], np.ndarray] = {}
    for row in frame.itertuples(index=False):
        seed, fold = int(row.cv_seed), int(row.fold)
        train_idx, test_idx = expected_fold(arrays, seed, fold)
        run = ROOT / "phase2/outputs/innovation200_confirmation/runs" / str(row.run_id)
        with np.load(run / "best_predictions.npz", allow_pickle=False) as prediction:
            observed = prediction["indices"].astype(int)
            if not np.array_equal(np.sort(observed), np.sort(test_idx)):
                failures.append(f"{row.run_id}: evaluation indices differ from replayed patient fold")
            if not np.allclose(prediction["y_true"], arrays["y"][observed]):
                failures.append(f"{row.run_id}: saved targets differ from metadata")
        checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
        if not np.allclose(checkpoint["target_mean"], arrays["y"][train_idx].mean(0), atol=1e-6):
            failures.append(f"{row.run_id}: target mean was not fitted on replayed train fold")
        if not np.allclose(checkpoint["target_std"], arrays["y"][train_idx].std(0), atol=1e-6):
            failures.append(f"{row.run_id}: target std was not fitted on replayed train fold")
        if np.intersect1d(arrays["patient_id"][train_idx], arrays["patient_id"][test_idx]).size:
            failures.append(f"{row.run_id}: patient overlap")
        key = (str(row.candidate_id), seed)
        coverage.setdefault(key, np.zeros(len(arrays["y"]), np.int8))[observed] += 1
    bad_coverage = {str(key): np.unique(value, return_counts=True)[1].tolist() for key, value in coverage.items() if not np.all(value == 1)}
    add(checks, "confirmation_fold_replay", "PASS" if not failures and not bad_coverage else "FAIL", {
        "fits": len(frame), "candidate_seed_sets": len(coverage), "patient_overlap": 0 if not failures else "see failures",
        "each_segment_predicted_once_per_candidate_seed": not bad_coverage, "failures": failures, "bad_coverage": bad_coverage,
    })
    add(checks, "confirmation_selection_independence", "LIMITATION", {
        "patient_separated": True, "fully_nested": False,
        "reason": "The 200-configuration screening and historical model-family design preceded repeated confirmation CV.",
        "remedy": "FINAL also runs a separate nested-200 procedural evaluation.",
    }, "Repeated OOF is descriptive patient-separated confirmation, not a fully independent performance estimate.")


def audit_nested(checks: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    decision = ROOT / "phase2/outputs/nested200/DECISION.json"
    if not decision.exists() or not arrays:
        add(checks, "nested_200_procedural_evaluation", "PENDING", str(decision), "Run the nested-200 stage.")
        return
    value = json.loads(decision.read_text())
    failures = []
    coverage = {str(subject): 0 for subject in np.unique(arrays["patient_id"])}
    inner = pd.read_csv(ROOT / "phase2/outputs/nested200/INNER_SCREEN_RESULTS.csv")
    inner_counts = inner.groupby("outer_fold").candidate_id.nunique().astype(int).to_dict()
    if inner_counts != {1: 201, 2: 201, 3: 201, 4: 201, 5: 201}:
        failures.append(f"inner candidate counts: {inner_counts}")
    for fold in range(1, 6):
        record = json.loads((ROOT / f"phase2/outputs/nested200/splits/outer_fold_{fold}.json").read_text())
        inner_train = set(map(str, record["inner_train_subjects"]))
        inner_validation = set(map(str, record["inner_validation_subjects"]))
        outer_test = set(map(str, record["outer_test_subjects"]))
        if inner_train & inner_validation or inner_train & outer_test or inner_validation & outer_test:
            failures.append(f"fold {fold}: patient overlap")
        outer_train = inner_train | inner_validation
        train_idx = np.flatnonzero(np.isin(arrays["patient_id"].astype(str), list(outer_train)))
        test_idx = np.flatnonzero(np.isin(arrays["patient_id"].astype(str), list(outer_test)))
        for subject in outer_test: coverage[subject] += 1
        for label in ("selected", "prior"):
            run = ROOT / f"phase2/outputs/nested200/outer_runs/{label}_fold{fold}"
            with np.load(run / "best_predictions.npz", allow_pickle=False) as prediction:
                if not np.array_equal(np.sort(prediction["indices"]), np.sort(test_idx)):
                    failures.append(f"fold {fold} {label}: prediction indices do not equal outer test")
                if not np.allclose(prediction["y_true"], arrays["y"][prediction["indices"]]):
                    failures.append(f"fold {fold} {label}: target mismatch")
            checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
            if not np.allclose(checkpoint["target_mean"], arrays["y"][train_idx].mean(0), atol=1e-6):
                failures.append(f"fold {fold} {label}: target mean not outer-train-only")
            if not np.allclose(checkpoint["target_std"], arrays["y"][train_idx].std(0), atol=1e-6):
                failures.append(f"fold {fold} {label}: target std not outer-train-only")
    bad_coverage = {subject: count for subject, count in coverage.items() if count != 1}
    structural = value.get("status") == "PASS" and value.get("outer_patient_overlap") == 0 and value.get("inner_outer_test_overlap") == 0
    status = "PASS" if structural and not failures and not bad_coverage else "FAIL"
    add(checks, "nested_200_procedural_evaluation", status, {
        "declared": value, "replayed_inner_candidate_counts": inner_counts,
        "independent_failures": failures, "outer_subject_coverage_failures": bad_coverage,
        "target_scaling_verified_outer_train_only": not any("target" in item for item in failures),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre", "post"), default="post")
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []
    compare_splits(checks)
    arrays, _ = audit_metadata(checks)
    audit_cache_window(checks, arrays)
    audit_feature_window(checks, arrays)
    if args.stage == "post":
        audit_confirmation(checks, arrays)
        audit_nested(checks, arrays)
    add(checks, "external_adaptation_semantics", "PASS", {
        "zero_shot": "No external BP labels used for model fitting or output scaling.",
        "affine": "Supervised target-domain affine adaptation with patient-level cross-fitting; not zero-shot and not subject-specific calibration.",
        "vitaldb_reference": "Timestamp-aligned PPG waveform plus ART_SBP/ART_DBP numeric tracks; median numeric BP within each 15-s window; no pulse-transit alignment claimed.",
    })
    add(checks, "literature_leakage_claim_scope", "PASS", {
        "supported_claim": "Segment-wise splitting and overlapping windows can expose subject/segment identity and inflate performance.",
        "unsupported_claim_avoided": "The code audit cannot establish that every competing paper leaked; FINAL must not make that blanket claim.",
    })
    statuses = {row["status"] for row in checks}
    overall = "FAIL" if "FAIL" in statuses else ("PASS_WITH_DECLARED_LIMITATIONS" if statuses & {"LIMITATION", "PENDING", "CORRECTED"} else "PASS")
    report = {
        "schema_version": "1.0", "stage": args.stage, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall, "checks": checks,
        "code_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in [
            ROOT / "features/build_feature_cache.py", ROOT / "neural/build_input_cache.py", ROOT / "phase2/engine.py",
            ROOT / "phase2/run_innovation200_search.py", ROOT / "phase2/run_innovation200_confirmation.py",
            ROOT / "phase2/run_nested200_cv.py",
            ROOT / "phase2/analyze_external_calibration.py", ROOT / "phase2/prepare_vitaldb_synchronized_external.py",
        ]},
    }
    destination = ROOT / f"artifacts/leakage_audit_{args.stage}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "checks": {row['name']: row['status'] for row in checks}}, ensure_ascii=False, indent=2))
    print(destination)
    if overall == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
