from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import load_arrays, run_experiment
from phase2.run_confirmation_cv import subject_strata


SEARCH = ROOT / "phase2/outputs/icbme_clinical_search"
OUT = ROOT / "phase2/outputs/icbme_clinical_confirmation"
LOCK = ROOT / "phase2/protocol/ICBME_CLINICAL_SEARCH_LOCK.json"
SEEDS = [42, 123, 2026]
BASELINE_METHOD = "resnet_multiview10__physio_multitask_aug"


def journal(event: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}, ensure_ascii=False) + "\n")


def functional_signature(config: dict) -> str:
    ignored = {"run_id", "stage", "clinical_method"}
    return json.dumps({key: config[key] for key in sorted(config) if key not in ignored}, sort_keys=True)


def select_candidates() -> list[dict]:
    ranking = pd.read_csv(SEARCH / "screening_ranking.csv")
    protocol = json.loads(LOCK.read_text(encoding="utf-8"))
    by_id = {config["run_id"]: config for config in protocol["configs"]}
    baseline_rows = ranking[ranking.clinical_method == BASELINE_METHOD]
    if baseline_rows.empty:
        raise RuntimeError("Fixed baseline is absent from completed screening")
    baseline_row = baseline_rows.sort_values("clinical_selection_score").iloc[0]
    baseline = dict(by_id[baseline_row.run_id])
    baseline["screening_epoch"] = int(baseline_row.epoch)
    baseline["selection_role"] = "fixed_baseline"
    gate = float(baseline_row.mean_mae) + 0.10
    selected = [baseline]
    signatures = {functional_signature(baseline)}
    for row in ranking.itertuples():
        config = dict(by_id[row.run_id])
        signature = functional_signature(config)
        if float(row.mean_mae) > gate or signature in signatures:
            continue
        config["screening_epoch"] = int(row.epoch)
        config["selection_role"] = "novel_candidate"
        selected.append(config)
        signatures.add(signature)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        raise RuntimeError("Fewer than two distinct novel candidates passed the safety gate")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SELECTED_CANDIDATES.json").write_text(json.dumps({
        "selected_after_screening": True,
        "baseline_mean_mae": float(baseline_row.mean_mae),
        "safety_gate_mean_mae": gate,
        "candidates": selected,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def main() -> None:
    arrays = load_arrays()
    subjects, strata = subject_strata(arrays["patient_id"], arrays["y"])
    candidates = select_candidates()
    rows_path = OUT / "fold_results.csv"
    rows = pd.read_csv(rows_path).to_dict("records") if rows_path.exists() else []
    completed = {(row["candidate_id"], int(row["seed"]), int(row["fold"])) for row in rows if row.get("status", "ok") == "ok"}
    journal("confirmation_started", candidates=[c["run_id"] for c in candidates], seeds=SEEDS, folds=5)
    for seed in SEEDS:
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (train_subject_i, test_subject_i) in enumerate(splitter.split(subjects, strata), 1):
            train_subjects, test_subjects = subjects[train_subject_i], subjects[test_subject_i]
            train_idx = np.flatnonzero(np.isin(arrays["patient_id"], train_subjects))
            test_idx = np.flatnonzero(np.isin(arrays["patient_id"], test_subjects))
            if np.intersect1d(train_subjects, test_subjects).size:
                raise RuntimeError("Patient leakage in confirmation fold")
            for number, base in enumerate(candidates, 1):
                key = (base["run_id"], seed, fold)
                if key in completed:
                    continue
                config = dict(base)
                config.update({
                    "run_id": f"ICBME_C{number}_seed{seed}_fold{fold}",
                    "stage": "ICBME_CONFIRM",
                    "epochs": max(3, int(base["screening_epoch"])),
                    "fixed_epochs": True,
                })
                journal("fold_started", candidate_id=base["run_id"], run_id=config["run_id"], seed=seed, fold=fold, train_subjects=len(train_subjects), test_subjects=len(test_subjects))
                result = run_experiment(config, arrays, train_idx, test_idx, OUT / "runs" / config["run_id"], seed=seed)
                result.update({
                    "candidate_id": base["run_id"], "clinical_method": base["clinical_method"],
                    "selection_role": base["selection_role"], "candidate_number": number,
                    "fold": fold, "status": "ok",
                })
                rows.append(result)
                pd.DataFrame(rows).to_csv(rows_path, index=False)
                journal("fold_completed", candidate_id=base["run_id"], run_id=config["run_id"], seed=seed, fold=fold, result=result)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["candidate_id", "clinical_method", "selection_role"], as_index=False).agg(
        fits=("selection_score", "size"),
        mae_sbp_mean=("mae_sbp", "mean"), mae_sbp_sd=("mae_sbp", "std"),
        mae_dbp_mean=("mae_dbp", "mean"), mae_dbp_sd=("mae_dbp", "std"),
        tail_mae_sbp_mean=("tail_mae_sbp", "mean"), tail_mae_dbp_mean=("tail_mae_dbp", "mean"),
        high_bp_sensitivity_mean=("high_bp_sensitivity", "mean"),
        high_bp_specificity_mean=("high_bp_specificity", "mean"),
        clinical_selection_score_mean=("clinical_selection_score", "mean"),
        clinical_selection_score_sd=("clinical_selection_score", "std"),
    ).sort_values("clinical_selection_score_mean")
    summary.to_csv(OUT / "summary.csv", index=False)
    journal("confirmation_completed", summary=summary.to_dict("records"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
