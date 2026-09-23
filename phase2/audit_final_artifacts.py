from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "phase2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 5e-5) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"Expected {expected}, observed {actual}")


def main() -> None:
    checks: list[dict] = []

    inventory = pd.read_csv(P2 / "literature/outputs/corpus_inventory.csv")
    unique_hashes = inventory["sha256"].nunique()
    extracted = int((inventory["extraction_status"] == "ok").sum()) if "extraction_status" in inventory else unique_hashes
    evidence_cards = len(list((P2 / "literature/outputs/evidence_cards").glob("*.md")))
    assert unique_hashes == 115 and evidence_cards == 87
    checks.append({"name": "literature", "unique_pdfs": unique_hashes, "full_text_extracted": extracted, "evidence_cards": evidence_cards})

    search = pd.read_csv(P2 / "outputs/search_results.csv")
    assert len(search) == 65 and search["run_id"].nunique() == 65
    assert search["selection_score"].notna().all()
    checks.append({"name": "locked_search", "successful_unique_configurations": 65})

    foundation = pd.read_csv(P2 / "outputs/foundation_adaptation/results.csv")
    assert len(foundation) == 5 and foundation["run_id"].nunique() == 5
    checks.append({"name": "foundation_adaptation", "exploratory_configurations": 5})

    folds = pd.read_csv(P2 / "outputs/confirmation_cv/fold_results.csv")
    assert len(folds) == 45
    assert set(folds.groupby("candidate_id").size()) == {15}
    assert set(folds["seed"].astype(int)) == {42, 123, 2026}
    assert set(folds["fold"].astype(int)) == {1, 2, 3, 4, 5}
    checks.append({"name": "confirmation", "fits": 45, "candidates": 3, "folds": 5, "seeds": 3})

    oof = pd.read_csv(P2 / "outputs/analysis/oof_ensemble_metrics.csv")
    ensemble = oof.loc[oof["model"] == "simple_mean_all_3"].iloc[0]
    close(float(ensemble.mae_sbp), 12.89865493774414)
    close(float(ensemble.mae_dbp), 8.18384075164795)
    close(float(ensemble.rmse_sbp), 16.284175872802734)
    close(float(ensemble.rmse_dbp), 10.774214744567871)
    checks.append({"name": "internal_ensemble", "mae_sbp": float(ensemble.mae_sbp), "mae_dbp": float(ensemble.mae_dbp)})

    external = pd.read_csv(P2 / "outputs/external_validation/external_metrics.csv")
    assert len(external) == 4 and external["ensemble_members"].eq(15).all()
    ppgbp = external.query("model == 'compatible_2s' and dataset == 'ppgbp219'").iloc[0]
    close(float(ppgbp.mae_sbp), 16.779277801513672)
    close(float(ppgbp.mae_dbp), 11.080109596252441)
    checks.append({"name": "external_zero_shot", "evaluations": 4, "ensemble_members_each": 15})

    calibration = pd.read_csv(P2 / "outputs/external_validation/calibration/crossfit_calibration_metrics.csv")
    assert len(calibration) == 16
    papagei = calibration.query("dataset_model == 'papagei10s_ambulatory56' and method == 'affine_ridge'").iloc[0]
    close(float(papagei.mae_sbp), 10.548460006713867)
    close(float(papagei.mae_dbp), 7.442013263702393)
    checks.append({"name": "external_crossfit", "rows": 16, "repeats": 10, "folds": 5})

    prepared = json.loads((P2 / "external/prepared/EXTERNAL_DATA_MANIFEST.json").read_text(encoding="utf-8"))
    assert prepared["ppg_bp"]["subjects"] == 219
    assert prepared["ppg_ambulatory_56"]["subjects"] == 56
    checks.append({"name": "external_data", "ppg_bp_subjects": 219, "ambulatory_subjects": 56})

    pdf = P2 / "paper/main_anonymous.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    pages = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")).strip())
    assert pages == 4
    log = (P2 / "paper/main_anonymous.log").read_text(encoding="utf-8", errors="replace")
    lower_log = log.lower()
    assert "citation" not in lower_log or "undefined" not in lower_log[lower_log.find("citation") : lower_log.find("citation") + 160]
    assert "there were undefined references" not in lower_log
    checks.append({"name": "paper", "pages": pages, "sha256": sha256(pdf), "undefined_references": 0})

    result = {
        "status": "PASS",
        "phase2_training_jobs": 65 + 5 + 45 + 45,
        "phase1_classical_pipelines": 200,
        "checks": checks,
        "important_interpretation": [
            "Confirmation is patient-separated repeated grouped CV but not fully nested after finalist screening.",
            "The prior Phase-1 test partition had already been inspected and is not described as untouched.",
            "External cuff labels differ from simultaneous invasive MIMIC-BP labels.",
            "External affine calibration is supervised and fully cross-fitted; it is not zero-shot.",
        ],
    }
    out = P2 / "reports/FINAL_AUDIT.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
