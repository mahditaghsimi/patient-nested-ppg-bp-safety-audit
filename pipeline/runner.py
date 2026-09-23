from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
LOGS = ROOT / "logs"
STATE_PATH = ARTIFACTS / "pipeline_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nonempty(relative: str, minimum_bytes: int = 1) -> bool:
    path = ROOT / relative
    return path.is_file() and path.stat().st_size >= minimum_bytes


def json_has(relative: str, *keys: str) -> bool:
    path = ROOT / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return all(key in value for key in keys)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def json_pass(relative: str) -> bool:
    path = ROOT / relative
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") in {
            "PASS", "PASS_WITH_DECLARED_LIMITATIONS"
        }
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return False


def csv_rows(relative: str, expected: int) -> bool:
    try:
        import pandas as pd

        return len(pd.read_csv(ROOT / relative)) == expected
    except Exception:
        return False


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = []
    for pattern in ("*.py", "*.yaml", "*.toml"):
        paths.extend(ROOT.rglob(pattern))
    # Search definitions are source inputs. Confirmation locks are generated
    # from fresh screening results and must not invalidate resume by changing
    # the pipeline fingerprint after their own stage completes.
    for name in ("PROTOCOL_LOCK.json", "ICBME_CLINICAL_SEARCH_LOCK.json", "ICBME_EVENT_SEARCH_LOCK.json", "INNOVATION200_SEARCH_LOCK.json", "VITALDB_SYNCHRONIZED_EXTERNAL_LOCK.json"):
        path = ROOT / "phase2/protocol" / name
        if path.is_file():
            paths.append(path)
    for path in sorted({p for p in paths if "outputs" not in p.parts and "artifacts" not in p.parts}):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_state() -> dict:
    if not STATE_PATH.exists() or STATE_PATH.stat().st_size == 0:
        return {"schema_version": "1.0", "runs": [], "stages": {}}
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        backup = STATE_PATH.with_suffix(f".corrupt-{int(time.time())}.json")
        STATE_PATH.replace(backup)
        return {"schema_version": "1.0", "runs": [], "stages": {}, "recovered_from": backup.name}
    if not isinstance(value, dict):
        return {"schema_version": "1.0", "runs": [], "stages": {}}
    value.setdefault("runs", [])
    value.setdefault("stages", {})
    return value


def save_state(state: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    args: tuple[str, ...]
    complete: Callable[[], bool]
    group: str = "compute"
    optional: bool = False


STAGES = (
    Stage("check_inputs", "scripts/check_inputs.py", (), lambda: json_pass("artifacts/input_audit.json")),
    Stage("official_split", "scripts/build_official_split.py", (), lambda: json_has("data/mimic_bp/splits/official_patient_split.json", "train", "validation", "test")),
    Stage("feature_cache", "features/build_feature_cache.py", (), lambda: nonempty("outputs/features/feature_cache.npz", 1024)),
    Stage("waveform_cache", "neural/build_input_cache.py", (), lambda: nonempty("outputs/neural/cache/ppg15s_3ch_float16.npy", 1024)),
    Stage("external_cuff_prepare", "phase2/prepare_external.py", (), lambda: json_has("data/external/prepared/EXTERNAL_DATA_MANIFEST.json", "ppg_bp", "ppg_ambulatory_56")),
    Stage("vitaldb_prepare", "phase2/prepare_vitaldb_synchronized_external.py", (), lambda: json_has("data/external/vitaldb/prepared/MANIFEST.json", "status")),
    Stage("clinical_sqi", "phase2/build_clinical_sqi.py", (), lambda: nonempty("phase2/outputs/clinical_sqi/clinical_sqi.npz", 1024)),
    Stage("leakage_pre", "scripts/audit_leakage.py", ("--stage", "pre"), lambda: json_pass("artifacts/leakage_audit_pre.json")),
    Stage("pilot_protocol", "phase2/registry.py", (), lambda: json_has("phase2/protocol/PROTOCOL_LOCK.json", "stage_A")),
    Stage("pilot_search", "phase2/run_search.py", (), lambda: csv_rows("phase2/outputs/search_results.csv", 65)),
    Stage("foundation", "phase2/run_foundation_adaptation.py", (), lambda: csv_rows("phase2/outputs/foundation_adaptation/results.csv", 5)),
    Stage("pilot_confirmation", "phase2/run_confirmation_cv.py", (), lambda: csv_rows("phase2/outputs/confirmation_cv/fold_results.csv", 45)),
    Stage("pilot_analysis", "phase2/analyze_confirmation.py", (), lambda: nonempty("phase2/outputs/analysis/oof_predictions.npz", 1024)),
    Stage("cuff_external_models", "phase2/run_external_validation.py", (), lambda: json_has("phase2/outputs/external_validation/EXTERNAL_RESULTS.json")),
    Stage("cuff_external_calibration", "phase2/analyze_external_calibration.py", (), lambda: nonempty("phase2/outputs/external_validation/calibration/crossfit_calibration_metrics.csv")),
    Stage("clinical_protocol", "phase2/icbme_clinical_registry.py", (), lambda: json_has("phase2/protocol/ICBME_CLINICAL_SEARCH_LOCK.json", "configs")),
    Stage("clinical_search", "phase2/run_icbme_clinical_search.py", (), lambda: csv_rows("phase2/outputs/icbme_clinical_search/screening_results.csv", 96)),
    Stage("clinical_confirmation", "phase2/run_icbme_clinical_confirmation.py", (), lambda: csv_rows("phase2/outputs/icbme_clinical_confirmation/fold_results.csv", 45)),
    Stage(
        "clinical_analysis",
        "phase2/analyze_icbme_clinical_results.py",
        (),
        lambda: json_has(
            "phase2/outputs/icbme_clinical_analysis/DECISION.json",
            "confirmed_primary_improvement",
        ),
    ),
    Stage("event_search", "phase2/run_icbme_event_search.py", (), lambda: csv_rows("phase2/outputs/icbme_event_search/screening_results.csv", 36)),
    Stage("event_safety", "phase2/analyze_icbme_medical_safety.py", (), lambda: json_has("phase2/outputs/icbme_medical_safety/MEDICAL_SAFETY_DECISION.json")),
    Stage("innovation_protocol", "phase2/innovation200_registry.py", (), lambda: json_has("phase2/protocol/INNOVATION200_SEARCH_LOCK.json", "configs")),
    Stage("innovation_reference", "phase2/run_innovation200_reference.py", (), lambda: json_has("phase2/outputs/innovation200_search/reference_prior_winner/REFERENCE_RESULT.json")),
    Stage("innovation_search", "phase2/run_innovation200_search.py", (), lambda: csv_rows("phase2/outputs/innovation200_search/screening_results.csv", 200)),
    Stage("innovation_screen_analysis", "phase2/analyze_innovation200_screening.py", (), lambda: nonempty("phase2/outputs/innovation200_search/METHOD_BY_METHOD.csv")),
    Stage("innovation_freeze", "phase2/select_innovation200_finalists.py", (), lambda: json_has("phase2/protocol/INNOVATION200_CONFIRMATION_LOCK.json", "finalists")),
    Stage("innovation_confirmation", "phase2/run_innovation200_confirmation.py", (), lambda: csv_rows("phase2/outputs/innovation200_confirmation/fold_results.csv", 75)),
    Stage("innovation_analysis", "phase2/analyze_innovation200_confirmation.py", (), lambda: json_has("phase2/outputs/innovation200_confirmation/DECISION.json", "status")),
    Stage("vitaldb_external", "phase2/evaluate_innovation200_vitaldb.py", (), lambda: json_has("phase2/outputs/innovation200_vitaldb_external/DECISION.json", "zero_shot")),
    Stage("innovation_finalize", "phase2/finalize_innovation200.py", (), lambda: json_pass("phase2/reports/INNOVATION200_AUTONOMOUS_AUDIT.json")),
    Stage("nested_200", "phase2/run_nested200_cv.py", (), lambda: json_has("phase2/outputs/nested200/DECISION.json", "summary", "event_safety")),
    Stage("leakage_post", "scripts/audit_leakage.py", ("--stage", "post"), lambda: json_pass("artifacts/leakage_audit_post.json")),
    Stage("data_manifest", "scripts/build_data_manifest.py", (), lambda: json_has("data/DATA_MANIFEST.json"), "report"),
    Stage("tables_figures", "scripts/build_final_tables_figures.py", (), lambda: json_has("results/tables/TABLE_INDEX.json", "tables") and json_has("results/figures/FIGURE_INDEX.json", "figures"), "report"),
    Stage("paper", "scripts/build_reviewer_corrected_paper.py", (), lambda: nonempty("paper/main_reviewer_corrected.pdf", 1024), "report"),
    Stage("code_manifest", "scripts/build_code_manifest.py", (), lambda: json_has("docs/CODE_MANIFEST.json"), "report"),
    Stage("repository_validation", "scripts/validate_repository.py", (), lambda: json_pass("artifacts/validation_report.json"), "audit"),
    Stage("claim_audit", "scripts/verify_claims.py", (), lambda: json_pass("artifacts/claim_audit.json"), "audit"),
    Stage("tests", "scripts/run_tests.py", (), lambda: False, "audit"),
    Stage("final_audit", "scripts/final_completion_audit.py", (), lambda: json_pass("artifacts/FINAL_COMPLETION_AUDIT.json"), "audit"),
    Stage("summary", "scripts/print_final_summary.py", (), lambda: nonempty("artifacts/FINAL_SUMMARY.md"), "report"),
)


def environment_snapshot(arguments: list[str]) -> None:
    snapshot = {
        "created_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "arguments": arguments,
        "code_fingerprint": code_fingerprint(),
    }
    try:
        import numpy
        import pandas
        import scipy
        import sklearn
        import torch

        snapshot["packages"] = {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        }
        snapshot["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as error:
        snapshot["package_probe_error"] = repr(error)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "environment.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage(stage: Stage) -> dict:
    command = [sys.executable, str(ROOT / stage.script), *stage.args]
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{stage.name}.log"
    started = time.time()
    print(f"\n[{stage.name}] {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- {utc_now()} ---\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError(f"Could not capture output for {stage.name}")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Stage {stage.name} failed with exit code {return_code}; see {log_path}")
    if not stage.complete():
        raise RuntimeError(f"Stage {stage.name} exited successfully but its output contract failed")
    return {
        "status": "PASS",
        "finished_utc": utc_now(),
        "seconds": round(time.time() - started, 3),
        "script": stage.script,
        "args": list(stage.args),
        "log": str(log_path.relative_to(ROOT)),
    }


def select_stages(mode: str) -> list[Stage]:
    if mode == "compute":
        return [stage for stage in STAGES if stage.group == "compute"]
    if mode == "report":
        return [stage for stage in STAGES if stage.group == "report"]
    if mode == "audit":
        return [stage for stage in STAGES if stage.group == "audit"]
    return list(STAGES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the complete PPG-to-BP article pipeline in order.")
    parser.add_argument("--mode", choices=("all", "compute", "report", "audit"), default="all")
    parser.add_argument("--from-stage", choices=[stage.name for stage in STAGES])
    parser.add_argument("--to-stage", choices=[stage.name for stage in STAGES])
    parser.add_argument("--force", action="append", default=[], choices=[stage.name for stage in STAGES])
    parser.add_argument("--dry-run", action="store_true", help="Print the ordered plan without executing scripts.")
    parser.add_argument("--list-stages", action="store_true")
    args = parser.parse_args(argv)

    if args.list_stages:
        for index, stage in enumerate(STAGES, 1):
            print(f"{index:02d} {stage.group:7s} {stage.name}{' [optional]' if stage.optional else ''}")
        return 0

    selected = select_stages(args.mode)
    names = [stage.name for stage in selected]
    if args.from_stage:
        if args.from_stage not in names:
            parser.error("--from-stage is outside the selected mode")
        selected = selected[names.index(args.from_stage):]
    if args.to_stage:
        names = [stage.name for stage in selected]
        if args.to_stage not in names:
            parser.error("--to-stage is outside the selected range")
        selected = selected[: names.index(args.to_stage) + 1]
    if args.dry_run:
        for stage in selected:
            print(f"{stage.name}: {sys.executable} {stage.script} {' '.join(stage.args)}".rstrip())
        return 0

    environment_snapshot(sys.argv if argv is None else argv)
    state = load_state()
    fingerprint = code_fingerprint()
    run_record = {"started_utc": utc_now(), "mode": args.mode, "fingerprint": fingerprint}
    state["runs"].append(run_record)
    save_state(state)

    try:
        for stage in selected:
            previous = state["stages"].get(stage.name, {})
            can_resume = (
                stage.name not in args.force
                and previous.get("status") == "PASS"
                and previous.get("fingerprint") == fingerprint
                and stage.complete()
            )
            if can_resume:
                print(f"SKIP verified: {stage.name}", flush=True)
                continue
            result = run_stage(stage)
            state["stages"][stage.name] = {**result, "fingerprint": fingerprint}
            save_state(state)
    except Exception as error:
        run_record.update({"status": "FAIL", "finished_utc": utc_now(), "error": repr(error)})
        save_state(state)
        raise

    run_record.update({"status": "PASS", "finished_utc": utc_now()})
    save_state(state)
    print(f"\nPIPELINE PASS: {STATE_PATH}", flush=True)
    return 0
