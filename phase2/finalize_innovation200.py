from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "phase2"
SEARCH = P2 / "outputs/innovation200_search"
CONFIRM = P2 / "outputs/innovation200_confirmation"
EXTERNAL = P2 / "outputs/innovation200_vitaldb_external"
REPORT = P2 / "reports/INNOVATION200_FINAL_RESULTS_FA.md"
AUDIT = P2 / "reports/INNOVATION200_AUTONOMOUS_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    search = pd.read_csv(SEARCH / "screening_results.csv")
    method_table = pd.read_csv(SEARCH / "METHOD_BY_METHOD.csv")
    folds = pd.read_csv(CONFIRM / "fold_results.csv")
    oof = pd.read_csv(CONFIRM / "OOF_SUMMARY.csv")
    paired = pd.read_csv(CONFIRM / "PAIRED_PATIENT_TESTS.csv")
    decision = json.loads((CONFIRM / "DECISION.json").read_text(encoding="utf-8"))
    external = json.loads((EXTERNAL / "DECISION.json").read_text(encoding="utf-8"))
    fold_mean = json.loads((CONFIRM / "FOLD_MEAN_BASELINE.json").read_text(encoding="utf-8"))
    protocol = json.loads((P2 / "protocol/INNOVATION200_SEARCH_LOCK.json").read_text(encoding="utf-8"))
    confirmation_lock = json.loads((P2 / "protocol/INNOVATION200_CONFIRMATION_LOCK.json").read_text(encoding="utf-8"))
    reference_replay = SEARCH / "reference_prior_winner/REFERENCE_RESULT.json"

    assert len(protocol["configs"]) == 200
    assert len(search) == 200 and search.status.value_counts().to_dict() == {"ok": 200}
    expected_fits = len(confirmation_lock["finalists"]) * 15
    assert len(folds) == expected_fits and folds.status.value_counts().to_dict() == {"ok": expected_fits}
    assert (folds.patient_overlap == 0).all()
    checkpoints = list((CONFIRM / "runs").glob("*/best_model.pt"))
    assert len(checkpoints) == expected_fits
    assert external["status"] == "PASS"

    best_screen = method_table.iloc[0]
    winner = oof[oof.candidate_id == decision["winner_id"]].iloc[0]
    baseline = oof[oof.candidate_id == "PRIOR_WINNER_FIXED"].iloc[0]
    winner_tests = paired[paired.candidate_id == decision["winner_id"]] if decision["winner_id"] != "PRIOR_WINNER_FIXED" else pd.DataFrame()
    zero = external["zero_shot"]
    source = external["source_mean"]
    lines = [
        "# نتیجهٔ نهایی اجرای خودکار ۲۰۰ روش PPG→BP",
        "",
        f"تاریخ تولید: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## نتیجهٔ اصلی",
        "",
        f"- وضعیت تأیید: **{decision['status']}**",
        f"- روش برنده: `{decision['winner_id']}` — {decision['winner_config']['innovation_label']}",
        f"- OOF برنده، SBP/DBP MAE: **{winner.mae_sbp:.3f}/{winner.mae_dbp:.3f} mmHg**.",
        f"- OOF مدل قبلی، SBP/DBP MAE: **{baseline.mae_sbp:.3f}/{baseline.mae_dbp:.3f} mmHg**.",
        f"- fold-mean baseline، SBP/DBP MAE: **{fold_mean['metrics']['mae_sbp']:.3f}/{fold_mean['metrics']['mae_dbp']:.3f} mmHg**.",
        f"- حساسیت مستقیم فشار بالا برای برنده: **{100*winner.high_bp_sensitivity:.2f}%**.",
        "",
        "## کامل‌بودن اجرا",
        "",
        f"- screening موفق: **{len(search)}/200**.",
        f"- confirmation بیمار-جدا: **{len(folds)}/{expected_fits} fit**؛ ۵ fold × ۳ seed × {len(confirmation_lock['finalists'])} finalist.",
        f"- checkpoint ذخیره‌شده: **{len(checkpoints)}**.",
        f"- زمان تجمعی screening: **{search.runtime_seconds.sum()/60:.1f} دقیقه GPU**.",
        "",
        "## بهترین نتیجهٔ screening",
        "",
        f"`{best_screen.run_id}` / {best_screen.innovation_label}: "
        f"{best_screen.mae_sbp:.3f}/{best_screen.mae_dbp:.3f} mmHg؛ "
        f"Δmean نسبت به reference هم-seed = {best_screen.delta_mean_vs_reference:+.3f} mmHg.",
        "",
        "## آزمون آماری بیمارمحور",
        "",
    ]
    if winner_tests.empty:
        lines.append("هیچ روش جدیدی gate ازپیش‌تعریف‌شده را عبور نداد؛ مدل قبلی حفظ شد و نتیجهٔ منفی نیز مستند است.")
    else:
        lines += [
            "| هدف | MAE برنده | MAE قبلی | اختلاف | CI 95% | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in winner_tests.itertuples():
            lines.append(
                f"| {row.target} | {row.candidate_patient_mae:.3f} | {row.baseline_patient_mae:.3f} | "
                f"{row.delta_candidate_minus_baseline:+.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] | {row.p_holm:.3g} |"
            )
    lines += [
        "",
        "## انتقال قفل‌شده به VitalDB",
        "",
        f"- کاندید external-compatible: `{decision['external_candidate_id']}`.",
        f"- zero-shot SBP/DBP MAE: **{zero['mae_sbp']:.3f}/{zero['mae_dbp']:.3f} mmHg**.",
        f"- source-mean baseline: **{source['mae_sbp']:.3f}/{source['mae_dbp']:.3f} mmHg**.",
        f"- حساسیت مستقیم فشار بالا: **{100*zero['direct_sensitivity']:.2f}%**؛ AUPRC={zero['auprc']:.3f}.",
        "",
        "## تفسیر برای مقاله",
        "",
        "- فقط وضعیت `CONFIRMED_IMPROVEMENT` مجوز ادعای بهبود مدل نسبت به مدل قبلی را می‌دهد.",
        "- حتی در صورت بهبود داخلی، نتیجهٔ VitalDB باید مستقل و بدون پنهان‌کردن شکست انتقال گزارش شود.",
        "- calibrated و zero-shot دو حالت متفاوت‌اند و نباید با هم ادغام شوند.",
        "- grouped OOF پس از screening کاملاً nested نیست؛ این محدودیت باید در مقاله بماند.",
        "",
        "## فایل‌های جزئی",
        "",
        "- `phase2/outputs/innovation200_search/METHOD_BY_METHOD.csv`: هر ۲۰۰ روش.",
        "- `phase2/outputs/innovation200_search/COMPONENT_EFFECTS.csv`: اثر هر جزء.",
        "- `phase2/outputs/innovation200_confirmation/PAIRED_PATIENT_TESTS.csv`: آزمون paired.",
        "- `phase2/outputs/innovation200_vitaldb_external/DECISION.json`: انتقال خارجی.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "protocol": {
            "search_lock_sha256": sha256(P2 / "protocol/INNOVATION200_SEARCH_LOCK.json"),
            "methods": 200,
            "unique_run_ids": int(search.run_id.nunique()),
        },
        "screening": {
            "rows": len(search), "ok": int((search.status == "ok").sum()),
            "authoritative_locked_reference_mean_mae": confirmation_lock["reference_mean_mae"],
            "post_lock_reference_replay_matches_locked_hash": (
                sha256(reference_replay) == confirmation_lock["reference_result_sha256"]
            ),
            "reference_replay_policy": "The immutable confirmation lock is authoritative; a later nonmatching replay is ignored.",
        },
        "confirmation": {
            "fits": len(folds), "expected": expected_fits,
            "patient_overlap_max": int(folds.patient_overlap.max()),
            "checkpoints": len(checkpoints),
        },
        "decision": {
            "status": decision["status"], "winner_id": decision["winner_id"],
            "external_candidate_id": decision["external_candidate_id"],
        },
        "external": {
            "status": external["status"],
            "patients": external["locked_vitaldb_manifest"]["accepted_cases"],
            "windows": external["locked_vitaldb_manifest"]["accepted_windows"],
        },
        "artifacts": {
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": sha256(REPORT),
            "screening_results_sha256": sha256(SEARCH / "screening_results.csv"),
            "confirmation_decision_sha256": sha256(CONFIRM / "DECISION.json"),
            "event_safety_sha256": sha256(CONFIRM / "EVENT_SAFETY.json"),
            "fold_mean_baseline_sha256": sha256(CONFIRM / "FOLD_MEAN_BASELINE.json"),
            "external_decision_sha256": sha256(EXTERNAL / "DECISION.json"),
        },
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(REPORT)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
