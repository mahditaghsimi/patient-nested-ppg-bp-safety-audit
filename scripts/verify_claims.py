#!/usr/bin/env python3
"""Cross-check freshly rendered manuscript numbers against fresh artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    split = json.loads((ROOT / "data/mimic_bp/splits/official_patient_split.json").read_text(encoding="utf-8"))
    nested = json.loads((ROOT / "phase2/outputs/nested200/DECISION.json").read_text(encoding="utf-8"))
    repeated = json.loads((ROOT / "phase2/outputs/innovation200_confirmation/DECISION.json").read_text(encoding="utf-8"))
    vital = json.loads((ROOT / "phase2/outputs/innovation200_vitaldb_external/DECISION.json").read_text(encoding="utf-8"))
    manuscript = (ROOT / "paper/main_reviewer_corrected.tex").read_text(encoding="utf-8")
    selected = nested["summary"]["nested_selected_procedure"]
    prior = nested["summary"]["historical_fixed_prior"]
    mean = nested["summary"]["outer_fold_training_mean"]
    zero = vital["zero_shot"]
    source_mean = vital["source_mean"]
    adapted = vital["supervised_affine_context"]
    expected_text = {
        "official_split": "1,100/195/229",
        "segments": f"{split['counts']['retained_segments']:,}",
        "nested_mae": f"{selected['mae_sbp']:.2f}/{selected['mae_dbp']:.2f}",
        "prior_mae": f"{prior['mae_sbp']:.2f}/{prior['mae_dbp']:.2f}",
        "mean_mae": f"{mean['mae_sbp']:.2f}/{mean['mae_dbp']:.2f}",
        "repeated_mae": f"{repeated['winner_oof']['mae_sbp']:.2f}/{repeated['winner_oof']['mae_dbp']:.2f}",
        "vital_zero_shot": f"{zero['mae_sbp']:.2f}/{zero['mae_dbp']:.2f}",
        "vital_source_training_mean": f"{source_mean['mae_sbp']:.2f}/{source_mean['mae_dbp']:.2f}",
        "vital_adapted": f"{adapted['mae_sbp']:.2f}/{adapted['mae_dbp']:.2f}",
    }
    checks = [
        {"name": name, "status": "PASS" if value in manuscript else "FAIL", "expected_text": value}
        for name, value in expected_text.items()
    ]
    checks.extend([
        {"name": "nested_patient_separation", "status": "PASS" if nested.get("outer_patient_overlap") == 0 and nested.get("each_segment_predicted_once") else "FAIL"},
        {"name": "external_scope_declared", "status": "PASS" if "external_model_scope" in vital and "fixed prior retained" in manuscript else "FAIL"},
        {"name": "source_mean_scope", "status": "PASS" if vital.get("source_mean_scope") == "official_internal_train_partition_only" else "FAIL"},
    ])
    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    report = {"status": status, "checks": checks, "policy": "Fresh artifact-to-manuscript consistency; no historical numeric snapshot is treated as truth."}
    destination = ROOT / "artifacts/claim_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
