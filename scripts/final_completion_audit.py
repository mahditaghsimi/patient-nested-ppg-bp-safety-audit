#!/usr/bin/env python3
"""Fail-closed final audit of data, leakage controls, runs, tables, and manuscript."""
from __future__ import annotations

import hashlib
import json
import py_compile
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/FINAL_COMPLETION_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict] = []

    def check(name: str, condition: bool, evidence) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    split = json.loads((ROOT / "data/mimic_bp/splits/official_patient_split.json").read_text())
    check("official_split", [len(split[k]) for k in ("train", "validation", "test")] == [1100, 195, 229], split["counts"])
    groups = [set(split[k]) for k in ("train", "validation", "test")]
    check("official_split_disjoint", not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "zero overlap")
    check("duplicate_policy", split["counts"]["retained_segments"] == 45689 and len(split["excluded_segments"]) == 31, split["counts"])

    with np.load(ROOT / "outputs/neural/cache/metadata.npz", allow_pickle=False) as metadata:
        n = len(metadata["y"]); patients = np.unique(metadata["patient_id"]).size
        partition_patients = {name: np.unique(metadata["patient_id"][metadata["split"].astype(str) == name]).size for name in ("train", "validation", "test")}
    waveform = np.load(ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy", mmap_mode="r")
    manifest = json.loads((ROOT / "outputs/neural/cache/manifest.json").read_text())
    check("central_input_cache", waveform.shape == (45689, 3, 1875) and n == 45689 and patients == 1524 and manifest["source_window"].startswith("central 15 s"),
          {"shape": waveform.shape, "patients": patients, "partition_patients": partition_patients, "source": manifest["source_window"]})

    expected_csv = {
        "phase2/outputs/search_results.csv": 65,
        "phase2/outputs/foundation_adaptation/results.csv": 5,
        "phase2/outputs/confirmation_cv/fold_results.csv": 45,
        "phase2/outputs/icbme_clinical_search/screening_results.csv": 96,
        "phase2/outputs/icbme_clinical_confirmation/fold_results.csv": 45,
        "phase2/outputs/icbme_event_search/screening_results.csv": 36,
        "phase2/outputs/innovation200_search/screening_results.csv": 200,
        "phase2/outputs/innovation200_confirmation/fold_results.csv": 75,
        "phase2/outputs/nested200/INNER_SCREEN_RESULTS.csv": 1005,
        "phase2/outputs/nested200/OUTER_FOLD_RESULTS.csv": 5,
    }
    for relative, expected in expected_csv.items():
        actual = len(pd.read_csv(ROOT / relative))
        check(f"row_count::{relative}", actual == expected, {"actual": actual, "expected": expected})

    nested = json.loads((ROOT / "phase2/outputs/nested200/DECISION.json").read_text())
    check("nested_no_patient_leakage", nested["status"] == "PASS" and nested["outer_patient_overlap"] == 0 and nested["inner_outer_test_overlap"] == 0 and nested["each_segment_predicted_once"], nested)
    with np.load(ROOT / "phase2/outputs/nested200/OOF_PREDICTIONS.npz", allow_pickle=False) as oof:
        check("nested_oof_finite", len(oof["y_true"]) == 45689 and np.isfinite(oof["selected"]).all(), {key: oof[key].shape for key in oof.files})
    leakage = json.loads((ROOT / "artifacts/leakage_audit_post.json").read_text())
    failed_leakage = [row["name"] for row in leakage["checks"] if row["status"] == "FAIL"]
    check("post_run_leakage_audit", not failed_leakage, {"overall": leakage["status"], "failures": failed_leakage})
    claim_audit = json.loads((ROOT / "artifacts/claim_audit.json").read_text())
    check("manuscript_claim_audit", claim_audit["status"] == "PASS", claim_audit)

    vital = json.loads((ROOT / "data/external/vitaldb/prepared/MANIFEST.json").read_text())
    check("vitaldb_qc", vital["status"] == "PASS" and vital["accepted_cases"] >= 80 and vital["accepted_windows"] >= 800, vital)
    calibration_audit = json.loads((ROOT / "phase2/outputs/external_validation/calibration/CROSSFIT_FOLD_AUDIT.json").read_text())
    overlaps = sum(item["patient_overlap"] for rows in calibration_audit.values() for item in rows)
    check("external_calibration_patient_separation", overlaps == 0, {"folds": sum(map(len, calibration_audit.values())), "overlap": overlaps})
    vital_calibration_audit = json.loads((ROOT / "phase2/outputs/innovation200_vitaldb_external/DESTINATION_CROSSFIT_AUDIT.json").read_text())
    vital_overlap = sum(item["patient_overlap"] for item in vital_calibration_audit)
    check("vitaldb_adaptation_patient_separation", len(vital_calibration_audit) == 5 and vital_overlap == 0,
          {"folds": len(vital_calibration_audit), "overlap": vital_overlap})

    table_index = json.loads((ROOT / "results/tables/TABLE_INDEX.json").read_text())
    figure_index = json.loads((ROOT / "results/figures/FIGURE_INDEX.json").read_text())
    missing_tables = [item["file"] for item in table_index["tables"] if not (ROOT / item["file"]).is_file()]
    missing_figures = [item["png"] for item in figure_index["figures"] if not (ROOT / item["png"]).is_file()]
    check("tables_complete", len(table_index["tables"]) >= 12 and not missing_tables, {"count": len(table_index["tables"]), "missing": missing_tables})
    check("figures_complete", len(figure_index["figures"]) >= 4 and not missing_figures, {"count": len(figure_index["figures"]), "missing": missing_figures})

    paper_files = [ROOT / "paper/main_reviewer_corrected.tex", ROOT / "paper/main_reviewer_corrected.pdf",
                   ROOT / "paper/main_reviewer_corrected_editable.docx", ROOT / "paper/supplement_reviewer_corrected.pdf"]
    check("paper_files", all(path.stat().st_size > 1000 for path in paper_files), {path.name: path.stat().st_size for path in paper_files})
    info = subprocess.run(["pdfinfo", str(paper_files[1])], capture_output=True, text=True, check=False)
    pages_match = re.search(r"^Pages:\s+(\d+)", info.stdout, flags=re.M)
    pages = int(pages_match.group(1)) if pages_match else None
    check("paper_pdf_readable_and_page_limit", info.returncode == 0 and pages is not None and pages <= 8,
          {"pages": pages, "official_maximum": 8, "pdfinfo": info.stdout})
    manuscript = paper_files[0].read_text(encoding="utf-8")
    required_phrases = ["official subject-disjoint split is 1,100/195/229", "patient-nested", "not causal ablations",
                        "supervised target-domain adaptation", "combined domain shift", "not presented as measured mean arterial pressure"]
    check("reviewer_corrections_in_manuscript", all(phrase in manuscript for phrase in required_phrases),
          {phrase: phrase in manuscript for phrase in required_phrases})

    compile_errors = []
    for path in sorted(ROOT.rglob("*.py")):
        try: py_compile.compile(str(path), doraise=True)
        except Exception as error: compile_errors.append({"path": str(path.relative_to(ROOT)), "error": repr(error)})
    check("python_sources_compile", not compile_errors, compile_errors)

    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    artifacts = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in paper_files}
    report = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "status": status,
              "checks": checks, "paper_artifacts": artifacts, "claim_boundary": "Retrospective research; no clinical-device claim."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": {row['name']: row['status'] for row in checks}}, ensure_ascii=False, indent=2))
    if status != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
