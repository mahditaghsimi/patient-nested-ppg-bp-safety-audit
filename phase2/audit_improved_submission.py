from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "phase2"
OUT = P2 / "reports" / "IMPROVED_SUBMISSION_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    return int(next(line.split(":", 1)[1] for line in output.splitlines() if line.startswith("Pages:")).strip())


def close(actual: float, expected: float, tolerance: float = 1e-3) -> None:
    assert abs(float(actual) - expected) <= tolerance, (actual, expected, tolerance)


def main() -> None:
    protocol_path = P2 / "protocol" / "VITALDB_SYNCHRONIZED_EXTERNAL_LOCK.json"
    manifest_path = P2 / "external" / "datasets" / "vitaldb_sync128" / "prepared" / "MANIFEST.json"
    prepared_path = P2 / "external" / "datasets" / "vitaldb_sync128" / "prepared" / "vitaldb_sync128.npz"
    decision_path = P2 / "outputs" / "vitaldb_synchronized_external" / "DECISION.json"
    paired_external_path = P2 / "outputs" / "vitaldb_synchronized_external" / "PAIRED_PATIENT_TESTS.csv"
    paired_internal_path = P2 / "outputs" / "analysis" / "ensemble_vs_population_mean_patient_tests.csv"
    inventory_path = P2 / "literature" / "outputs" / "corpus_inventory.csv"
    report_path = P2 / "reports" / "PAPER_IMPROVEMENT_LITERATURE_AUDIT_FA.md"
    tex_path = P2 / "paper" / "main_anonymous_medical.tex"
    pdf_path = P2 / "paper" / "main_anonymous_medical.pdf"
    word_path = P2 / "paper" / "main_anonymous_medical_editable.docx"

    required = [
        protocol_path, manifest_path, prepared_path, decision_path,
        paired_external_path, paired_internal_path, inventory_path,
        report_path, tex_path, pdf_path, word_path,
    ]
    for path in required:
        assert path.exists() and path.stat().st_size > 0, path

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert protocol["locked_before_downloading_waveform_outcomes"] is True
    assert protocol["selection"]["outcome_blind"] is True
    assert protocol["selection"]["requested_cases"] == 128
    assert manifest["status"] == "PASS"
    assert manifest["accepted_cases"] == 119
    assert manifest["accepted_windows"] == 2380
    assert manifest["prepared_sha256"] == sha256(prepared_path)

    primary = decision["primary_zero_shot"]
    baseline = decision["source_mean_baseline"]
    calibrated = decision["supervised_context"]["affine_calibration"]
    threshold = decision["supervised_context"]["event_threshold"]
    close(primary["mae_sbp"], 18.5666809082)
    close(primary["mae_dbp"], 11.5604305267)
    close(baseline["mae_sbp"], 17.8592548370)
    close(baseline["mae_dbp"], 10.4078617096)
    close(calibrated["mae_sbp"], 16.2825469971)
    close(calibrated["mae_dbp"], 9.2089424133)
    assert primary["direct_sensitivity"] == 0.0
    assert 0.30 < threshold["sensitivity"] < 0.31
    assert 0.89 < threshold["specificity"] < 0.91

    paired_external = pd.read_csv(paired_external_path)
    ext = paired_external[
        (paired_external["candidate"] == "resnet_zero_shot")
        & (paired_external["baseline"] == "mimic_source_mean")
    ].set_index("target")
    assert ext.loc["SBP", "mean_delta_candidate_minus_baseline"] > 0
    assert ext.loc["DBP", "mean_delta_candidate_minus_baseline"] > 0
    assert ext.loc["SBP", "p_holm"] < 0.01
    assert ext.loc["DBP", "p_holm"] < 1e-5

    paired_internal = pd.read_csv(paired_internal_path).set_index("target")
    close(paired_internal.loc["SBP", "delta"], -1.4777784)
    close(paired_internal.loc["DBP", "delta"], -1.441015)
    assert paired_internal.loc["SBP", "ci_high"] < 0
    assert paired_internal.loc["DBP", "ci_high"] < 0
    assert paired_internal.loc["SBP", "wilcoxon_p"] < 1e-39
    assert paired_internal.loc["DBP", "wilcoxon_p"] < 1e-39

    inventory = pd.read_csv(inventory_path)
    assert len(inventory) == 115
    assert (inventory["status"] == "ok").sum() == 115
    assert (inventory["relevance_score"] >= 20).sum() == 79

    tex = tex_path.read_text(encoding="utf-8")
    for marker in (
        "197 locked regression/event screens", "14.38/9.63",
        "synchronized VitalDB", "18.57/11.56", "direct high-BP sensitivity was 0",
        "kim2026", "lee2022", "must not be used for diagnosis",
    ):
        assert marker in tex, marker
    assert "TODO" not in tex and "PLACEHOLDER" not in tex
    assert pdf_pages(pdf_path) == 4

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS",
        "literature": {
            "unique_full_text_pdfs": len(inventory),
            "relevant_score_ge_20": int((inventory["relevance_score"] >= 20).sum()),
            "report": str(report_path.relative_to(ROOT)),
        },
        "vitaldb": {
            "protocol_locked_before_outcomes": True,
            "selected_cases": 128,
            "accepted_patients": 119,
            "accepted_windows": 2380,
            "prepared_sha256": sha256(prepared_path),
            "zero_shot_mae_sbp_dbp": [primary["mae_sbp"], primary["mae_dbp"]],
            "source_mean_mae_sbp_dbp": [baseline["mae_sbp"], baseline["mae_dbp"]],
            "direct_high_bp_sensitivity": primary["direct_sensitivity"],
        },
        "internal_baseline": {
            "ensemble_minus_mean_sbp": paired_internal.loc["SBP", "delta"],
            "ensemble_minus_mean_dbp": paired_internal.loc["DBP", "delta"],
        },
        "paper": {
            "pdf_pages": pdf_pages(pdf_path),
            "pdf_sha256": sha256(pdf_path),
            "tex_sha256": sha256(tex_path),
            "word_sha256": sha256(word_path),
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
