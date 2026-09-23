from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.evaluate_vitaldb_synchronized_external import (
    destination_crossfit, event_metrics, macro_metrics, paired_tests, patient_bootstrap,
)
from phase2.run_external_validation import infer_checkpoint


P2 = ROOT / "phase2"
DATA = ROOT / "data/external/vitaldb/prepared/vitaldb_sync128.npz"
MANIFEST = ROOT / "data/external/vitaldb/prepared/MANIFEST.json"
LOCK = P2 / "protocol/INNOVATION200_CONFIRMATION_LOCK.json"
DECISION = P2 / "outputs/innovation200_confirmation/DECISION.json"
RUNS = P2 / "outputs/innovation200_confirmation/runs"
INTERNAL_META = ROOT / "outputs/neural/cache/metadata.npz"
OUT = P2 / "outputs/innovation200_vitaldb_external"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise RuntimeError("VitalDB preparation gate failed")
    external = decision["external_candidate_config"]
    candidate_id = decision["external_candidate_id"]
    candidates = lock["finalists"]
    by_id = {row["run_id"]: (index, row) for index, row in enumerate(candidates, 1)}
    member_ids = external.get("members", [candidate_id])
    for member_id in member_ids:
        if by_id[member_id][1].get("engineered", "none") != "none":
            raise RuntimeError(f"External inference cannot reproduce engineered features: {member_id}")
    seeds = lock["confirmation"]["fresh_seeds"]
    with np.load(DATA, allow_pickle=False) as data:
        x, y, patient = data["x"].copy(), data["y"].copy(), data["patient_id"].copy()

    per_candidate = []
    all_checkpoint_predictions = []
    checkpoint_paths = []
    for member_id in member_ids:
        number, _ = by_id[member_id]
        member_predictions = []
        for seed in seeds:
            for fold in range(1, 6):
                checkpoint = RUNS / f"I200C_C{number}_seed{seed}_fold{fold}" / "best_model.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                prediction = infer_checkpoint(checkpoint, x)
                member_predictions.append(prediction)
                all_checkpoint_predictions.append(prediction)
                checkpoint_paths.append(str(checkpoint.relative_to(ROOT)))
        per_candidate.append(np.mean(member_predictions, axis=0))
    zero_shot = np.mean(per_candidate, axis=0).astype(np.float32)
    with np.load(INTERNAL_META, allow_pickle=False) as source:
        train_mask = source["split"].astype(str) == "train"
        source_training_mean = source["y"][train_mask].mean(axis=0)
        source_mean = np.broadcast_to(source_training_mean, y.shape).copy()
    calibrated, destination_mean, crossfit_event_prediction, crossfit_audit = destination_crossfit(patient, y, zero_shot)
    prediction_sets = {
        "mimic_source_mean": source_mean,
        "innovation200_zero_shot": zero_shot,
        "destination_mean_crossfit": destination_mean,
        "innovation200_affine_crossfit": calibrated,
    }
    rows = []
    for name, prediction in prediction_sets.items():
        rows.append({
            "method": name, **macro_metrics(patient, y, prediction),
            **patient_bootstrap(patient, y, prediction), **event_metrics(y, prediction),
        })
    summary = pd.DataFrame(rows)
    tests = pd.DataFrame(paired_tests(patient, y, prediction_sets, "mimic_source_mean"))
    event = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    tp = np.sum(event & crossfit_event_prediction); fn = np.sum(event & ~crossfit_event_prediction)
    tn = np.sum(~event & ~crossfit_event_prediction); fp = np.sum(~event & crossfit_event_prediction)
    threshold = {
        "method": "patient-cross-fitted threshold on unchanged external score",
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }
    zero_row = summary[summary.method == "innovation200_zero_shot"].iloc[0].to_dict()
    base_row = summary[summary.method == "mimic_source_mean"].iloc[0].to_dict()
    calibrated_row = summary[summary.method == "innovation200_affine_crossfit"].iloc[0].to_dict()
    previous_path = P2 / "outputs/vitaldb_synchronized_external/DECISION.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))["primary_zero_shot"] if previous_path.exists() else None
    output = {
        "status": "PASS",
        "locked_vitaldb_manifest": manifest,
        "internal_decision_status": decision["status"],
        "candidate_id": candidate_id,
        "external_model_scope": (
            "The external model is the locked secondary-confirmation candidate, "
            "not the fold-varying nested-selection procedure."
        ),
        "candidate_config": external,
        "component_candidate_ids": member_ids,
        "checkpoint_count": len(checkpoint_paths),
        "zero_shot": zero_row,
        "source_mean": base_row,
        "source_mean_scope": "official_internal_train_partition_only",
        "supervised_affine_context": calibrated_row,
        "crossfit_event_threshold": threshold,
        "previous_fixed_resnet_zero_shot": previous,
        "claim_limit": "VitalDB is a locked synchronized retrospective stress test, not prospective clinical validation; affine calibration is supervised.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    tests.to_csv(OUT / "PAIRED_PATIENT_TESTS.csv", index=False)
    (OUT / "DECISION.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "CHECKPOINTS.json").write_text(json.dumps(checkpoint_paths, indent=2) + "\n", encoding="utf-8")
    (OUT / "DESTINATION_CROSSFIT_AUDIT.json").write_text(json.dumps(crossfit_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        OUT / "PREDICTIONS.npz", y_true=y, patient_id=patient, zero_shot=zero_shot,
        source_mean=source_mean, affine_crossfit=calibrated, destination_mean=destination_mean,
        checkpoint_predictions=np.asarray(all_checkpoint_predictions, dtype=np.float32),
    )
    print(summary.to_string(index=False))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
