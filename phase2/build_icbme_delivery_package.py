from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "phase2"
DEST = P2 / "delivery" / "ICBME2026_PPG_BP_INNOVATION200_FINAL_20260821"
ZIP_PATH = DEST.parent / f"{DEST.name}.zip"


FILES = [
    # Final manuscript artifacts.
    "phase2/paper/main_anonymous_medical.pdf",
    "phase2/paper/main_anonymous_medical.tex",
    "phase2/paper/main_anonymous_medical_editable.docx",
    "phase2/paper/IEEEtran.cls",
    "phase2/paper/figures/pipeline_overview.png",
    "phase2/paper/AUTHOR_FIELDS_NEEDED_FA.md",
    "phase2/paper/build_word_draft.py",
    # Final narrative and machine-readable audit.
    "phase2/reports/ICBME2026_FINAL_SPRINT_RESULTS_FA.md",
    "phase2/reports/ICBME2026_FINAL_SPRINT_AUDIT.json",
    "phase2/reports/IMPROVED_SUBMISSION_AUDIT.json",
    "phase2/reports/PAPER_IMPROVEMENT_LITERATURE_AUDIT_FA.md",
    "phase2/reports/SUBMISSION_VALUE_DECISION_FA.md",
    "phase2/reports/MIMIC3_AUXILIARY_AND_FORECASTING_FINAL_FA.md",
    "phase2/reports/FORECASTING_VALUE_DECISION_FA.md",
    "phase2/reports/LITERATURE_REAUDIT_FA.md",
    "phase2/reports/PHASE2_RESULTS_FA.md",
    "phase2/reports/INNOVATION200_SCREENING_REPORT_FA.md",
    "phase2/reports/INNOVATION200_FINAL_RESULTS_FA.md",
    "phase2/reports/INNOVATION200_AUTONOMOUS_AUDIT.json",
    "phase2/reports/MODEL_LITERATURE_COVERAGE_FA.md",
    "phase2/reports/tables/all_65_search_runs.csv",
    "phase2/reports/tables/literature_gap_matrix.csv",
    "phase2/literature/outputs/corpus_inventory.csv",
    "phase2/literature/outputs/corpus_inventory.json",
    "phase2/literature/audit_full_corpus.py",
    # Locked protocols.
    "phase2/protocol/PROTOCOL_LOCK.json",
    "phase2/protocol/ICBME_CLINICAL_SEARCH_LOCK.json",
    "phase2/protocol/ICBME_EVENT_SEARCH_LOCK.json",
    "phase2/protocol/MIMIC3_AUX_PROTOCOL_LOCK.json",
    "phase2/protocol/MIMIC3_AUX_PROTOCOL_ADDENDUM_1.json",
    "phase2/protocol/VITALDB_SYNCHRONIZED_EXTERNAL_LOCK.json",
    "phase2/protocol/REPRODUCIBILITY_CONFIRMATION_LOCK_20260821.json",
    "phase2/protocol/INNOVATION200_SEARCH_LOCK.json",
    "phase2/protocol/INNOVATION200_CONFIRMATION_LOCK.json",
    # Reproduction code.
    "phase2/models.py",
    "phase2/engine.py",
    "phase2/registry.py",
    "phase2/icbme_clinical_registry.py",
    "phase2/build_clinical_sqi.py",
    "phase2/run_icbme_clinical_search.py",
    "phase2/run_icbme_clinical_confirmation.py",
    "phase2/analyze_icbme_clinical_results.py",
    "phase2/run_icbme_event_search.py",
    "phase2/analyze_icbme_medical_safety.py",
    "phase2/audit_icbme_final_sprint.py",
    "phase2/audit_improved_submission.py",
    "phase2/prepare_vitaldb_synchronized_external.py",
    "phase2/evaluate_vitaldb_synchronized_external.py",
    "phase2/run_reproducibility_confirmation.py",
    "phase2/run_icbme_clinical_pipeline.sh",
    "phase2/innovation200_registry.py",
    "phase2/run_innovation200_search.py",
    "phase2/run_innovation200_reference.py",
    "phase2/analyze_innovation200_screening.py",
    "phase2/select_innovation200_finalists.py",
    "phase2/run_innovation200_confirmation.py",
    "phase2/repair_innovation200_confirmation.py",
    "phase2/analyze_innovation200_confirmation.py",
    "phase2/evaluate_innovation200_vitaldb.py",
    "phase2/finalize_innovation200.py",
    "phase2/run_innovation200_pipeline.sh",
    "phase2/continue_innovation200_background.sh",
    # Compact result tables: no datasets, caches, checkpoints, or per-fit weights.
    "phase2/outputs/clinical_sqi/SQI_SUMMARY.csv",
    "phase2/outputs/clinical_sqi/SQI_MANIFEST.json",
    "phase2/outputs/icbme_clinical_search/screening_results.csv",
    "phase2/outputs/icbme_clinical_search/screening_ranking.csv",
    "phase2/outputs/icbme_clinical_confirmation/SELECTED_CANDIDATES.json",
    "phase2/outputs/icbme_clinical_confirmation/fold_results.csv",
    "phase2/outputs/icbme_clinical_confirmation/summary.csv",
    "phase2/outputs/icbme_clinical_analysis/OOF_CLINICAL_SUMMARY.csv",
    "phase2/outputs/icbme_clinical_analysis/SELECTIVE_PREDICTION.csv",
    "phase2/outputs/icbme_clinical_analysis/CONFORMAL_90.csv",
    "phase2/outputs/icbme_clinical_analysis/PAIRED_PATIENT_TESTS.csv",
    "phase2/outputs/icbme_clinical_analysis/DECISION.json",
    "phase2/outputs/icbme_event_search/screening_results.csv",
    "phase2/outputs/icbme_event_search/screening_ranking.csv",
    "phase2/outputs/icbme_medical_safety/MEDICAL_SAFETY_REPORT_FA.md",
    "phase2/outputs/icbme_medical_safety/MEDICAL_SAFETY_DECISION.json",
    "phase2/outputs/icbme_medical_safety/PRESSURE_STRATA.csv",
    "phase2/outputs/analysis/population_mean_baseline.csv",
    "phase2/outputs/analysis/ensemble_vs_population_mean_patient_tests.csv",
    "phase2/outputs/vitaldb_synchronized_external/DECISION.json",
    "phase2/outputs/vitaldb_synchronized_external/SUMMARY.csv",
    "phase2/outputs/vitaldb_synchronized_external/PAIRED_PATIENT_TESTS.csv",
    "phase2/outputs/vitaldb_synchronized_external/PAPAGEI_DIAGNOSTIC.json",
    "data/external/vitaldb/prepared/MANIFEST.json",
    # The complete 200-method and fresh-seed confirmation evidence.
    "phase2/outputs/innovation200_search/screening_results.csv",
    "phase2/outputs/innovation200_search/METHOD_BY_METHOD.csv",
    "phase2/outputs/innovation200_search/COMPONENT_EFFECTS.csv",
    "phase2/outputs/innovation200_search/STAGE_SUMMARY.csv",
    "phase2/outputs/innovation200_confirmation/fold_results.csv",
    "phase2/outputs/innovation200_confirmation/summary.csv",
    "phase2/outputs/innovation200_confirmation/OOF_SUMMARY.csv",
    "phase2/outputs/innovation200_confirmation/PAIRED_PATIENT_TESTS.csv",
    "phase2/outputs/innovation200_confirmation/DECISION.json",
    "phase2/outputs/innovation200_confirmation/FOLD_MEAN_BASELINE.json",
    "phase2/outputs/innovation200_confirmation/FOLD_MEAN_BASELINE.npz",
    "phase2/outputs/innovation200_confirmation/EVENT_SAFETY.json",
    "phase2/outputs/innovation200_confirmation/OOF_PREDICTIONS.npz",
    "phase2/outputs/innovation200_vitaldb_external/SUMMARY.csv",
    "phase2/outputs/innovation200_vitaldb_external/PAIRED_PATIENT_TESTS.csv",
    "phase2/outputs/innovation200_vitaldb_external/DECISION.json",
    "phase2/outputs/innovation200_vitaldb_external/PREDICTIONS.npz",
    "phase2/outputs/innovation200_vitaldb_external/CHECKPOINTS.json",
    # Official conference material retained locally.
    "phase2/conference_template/Writing_Guide_FA.docx",
    "phase2/conference_template/Paper_Template_2authors_EN.docx",
    "phase2/conference_template/submission_guide.html",
]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    selected = [ROOT / rel for rel in FILES]
    selected.extend(sorted((P2 / "paper" / "referenced_papers").glob("*.pdf")))
    selected.extend(sorted((P2 / "literature" / "outputs" / "evidence_cards").glob("*.md")))
    # Only candidate C1 was retained and used for the locked VitalDB test.
    # Its 15 fold/seed weights make the reported external inference executable.
    selected.extend(sorted((P2 / "outputs" / "innovation200_confirmation" / "runs").glob("I200C_C1_*/best_model.pt")))
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package inputs:\n" + "\n".join(missing))

    manifest_files = []
    for source in selected:
        relative = source.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest_files.append(
            {
                "path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": checksum(target),
            }
        )

    readme = DEST / "README_FA.txt"
    readme.write_text(
        "بسته بازتولید و ارسال مقاله ICBME2026\n"
        "======================================\n\n"
        "نسخه اصلی: phase2/paper/main_anonymous_medical.pdf\n"
        "نتیجه نهایی ۲۰۰ روش: phase2/reports/INNOVATION200_FINAL_RESULTS_FA.md\n"
        "ممیزی ماشینی: phase2/reports/INNOVATION200_AUTONOMOUS_AUDIT.json\n"
        "جدول هر ۲۰۰ روش: phase2/outputs/innovation200_search/METHOD_BY_METHOD.csv\n"
        "تصمیم آماری: phase2/outputs/innovation200_confirmation/DECISION.json\n"
        "نتیجه انتقال هم‌زمان: phase2/outputs/innovation200_vitaldb_external/DECISION.json\n"
        "اجرای زنجیره: phase2/run_innovation200_pipeline.sh\n\n"
        "دیتاست و cache سیگنال در بسته نیستند. ۱۵ checkpoint مدل نگه‌داشته‌شده که در VitalDB استفاده شدند موجودند؛ ۶۰ checkpoint نامزدهای ردشده حذف شده‌اند.\n"
        "پیش از ارسال نسخه نویسندگان، فایل AUTHOR_FIELDS_NEEDED_FA.md تکمیل شود.\n",
        encoding="utf-8",
    )
    manifest_files.append(
        {"path": "README_FA.txt", "bytes": readme.stat().st_size, "sha256": checksum(readme)}
    )
    manifest = {
        "package": DEST.name,
        "purpose": "ICBME2026 final 200-method PPG-to-BP clinical-safety manuscript and reproducibility bundle",
        "excluded": ["raw datasets", "signal caches", "60 rejected-candidate checkpoints"],
        "file_count": len(manifest_files),
        "files": sorted(manifest_files, key=lambda item: item["path"]),
    }
    manifest_path = DEST / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(DEST.name) / path.relative_to(DEST)))

    print(json.dumps({
        "directory": str(DEST),
        "zip": str(ZIP_PATH),
        "files": len(manifest_files) + 1,
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": checksum(ZIP_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
