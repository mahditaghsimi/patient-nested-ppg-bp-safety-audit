from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "phase2"
STAGE = P2 / "delivery_package"
ZIP = ROOT / "ICBME_PPG_BP_SUBMISSION_PACKAGE_2026-08-20.zip"
PACKAGE_ROOT = "ICBME_PPG_BP_SUBMISSION_PACKAGE_2026-08-20"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def selected_files() -> list[Path]:
    explicit = [
        P2 / "DELIVERY_README_FA.md",
        P2 / "__init__.py",
        P2 / "models.py",
        P2 / "engine.py",
        P2 / "registry.py",
        P2 / "prepare_external.py",
        P2 / "run_search.py",
        P2 / "run_foundation_adaptation.py",
        P2 / "run_confirmation_cv.py",
        P2 / "analyze_confirmation.py",
        P2 / "run_external_validation.py",
        P2 / "analyze_external_calibration.py",
        P2 / "audit_final_artifacts.py",
        P2 / "run_autonomous.sh",
        P2 / "protocol/PROTOCOL_LOCK.json",
        P2 / "external/prepared/EXTERNAL_DATA_MANIFEST.json",
        P2 / "paper/main_anonymous.tex",
        P2 / "paper/main_anonymous.pdf",
        P2 / "paper/main_anonymous_editable.docx",
        P2 / "paper/IEEEtran.cls",
        P2 / "paper/AUTHOR_FIELDS_NEEDED_FA.md",
        P2 / "paper/build_word_draft.py",
        P2 / "reports/PHASE2_RESULTS_FA.md",
        P2 / "reports/LITERATURE_REAUDIT_FA.md",
        P2 / "reports/FINAL_AUDIT.json",
        P2 / "reports/build_phase2_reports.py",
        P2 / "reports/build_paper_figure.py",
        P2 / "literature/audit_full_corpus.py",
        P2 / "literature/outputs/CORPUS_AUDIT.md",
        P2 / "literature/outputs/corpus_inventory.csv",
        P2 / "literature/outputs/corpus_inventory.json",
        P2 / "outputs/search_results.csv",
        P2 / "outputs/stage_c_resolved_registry.json",
        P2 / "outputs/selected_for_nested_cv.csv",
        P2 / "outputs/confirmation_cv/fold_results.csv",
        P2 / "outputs/confirmation_cv/summary.csv",
        P2 / "outputs/confirmation_cv/winner.json",
        P2 / "outputs/confirmation_cv/journal.jsonl",
        P2 / "outputs/analysis/oof_ensemble_metrics.csv",
        P2 / "outputs/analysis/paired_patient_tests.csv",
        P2 / "outputs/analysis/exploratory_ensemble_grid.csv",
        P2 / "outputs/analysis/seed_level_oof_metrics.csv",
        P2 / "outputs/foundation_adaptation/results.csv",
        P2 / "outputs/external_validation/external_metrics.csv",
        P2 / "outputs/external_validation/calibration/crossfit_calibration_metrics.csv",
    ]
    patterns = [
        "paper/figures/*",
        "paper/referenced_papers/*.pdf",
        "reports/figures/*.png",
        "reports/tables/*.csv",
        "literature/outputs/evidence_cards/*.md",
        "conference_template/*",
    ]
    files = list(explicit)
    for pattern in patterns:
        files.extend(path for path in P2.glob(pattern) if path.is_file())
    files = sorted(set(files))
    missing = [str(path.relative_to(ROOT)) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    return files


def main():
    if STAGE.exists() or ZIP.exists():
        raise FileExistsError("Delivery target already exists; refusing to overwrite it")
    STAGE.mkdir(parents=True)
    files = selected_files()
    records = []
    for source in files:
        relative = source.relative_to(P2)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({"path": str(relative), "bytes": target.stat().st_size, "sha256": digest(target)})

    manifest = {
        "package": PACKAGE_ROOT,
        "files": len(records),
        "total_uncompressed_bytes": sum(row["bytes"] for row in records),
        "contents": records,
        "excluded_on_purpose": [
            "raw and prepared datasets",
            "large trained fold checkpoints",
            "PaPaGei checkpoint and cloned repository",
            "per-run prediction arrays and temporary renderings",
        ],
        "reason_for_exclusion": "Keep the review/submission bundle compact; paths and data hashes remain in EXTERNAL_DATA_MANIFEST.json.",
    }
    manifest_path = STAGE / "PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sums = [f"{row['sha256']}  {row['path']}" for row in records]
    sums.append(f"{digest(manifest_path)}  PACKAGE_MANIFEST.json")
    (STAGE / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, Path(PACKAGE_ROOT) / path.relative_to(STAGE))
    print(json.dumps({"zip": str(ZIP), "bytes": ZIP.stat().st_size, "sha256": digest(ZIP), "files": len(records) + 2}, indent=2))


if __name__ == "__main__":
    main()
