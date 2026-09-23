from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import load_arrays, run_experiment
from phase2.run_confirmation_cv import subject_strata


LOCK = ROOT / "phase2/protocol/INNOVATION200_CONFIRMATION_LOCK.json"
OUT = ROOT / "phase2/outputs/innovation200_confirmation"


def journal(event: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}, ensure_ascii=False) + "\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    if not LOCK.exists():
        raise FileNotFoundError("Run select_innovation200_finalists.py after all 200 screens")
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    if not locked.get("locked_after_complete_screening_before_confirmation"):
        raise RuntimeError("Invalid confirmation lock")
    candidates = locked["finalists"]
    seeds = locked["confirmation"]["fresh_seeds"]
    arrays = load_arrays()
    subjects, strata = subject_strata(arrays["patient_id"], arrays["y"])
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "fold_results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    def cv_seed_of(row: dict) -> int:
        if "cv_seed" in row and not pd.isna(row["cv_seed"]):
            return int(row["cv_seed"])
        match = re.search(r"_seed(\d+)_fold\d+$", str(row["run_id"]))
        if not match:
            raise RuntimeError(f"Cannot recover CV seed from {row['run_id']}")
        return int(match.group(1))

    completed = {
        (str(row["candidate_id"]), cv_seed_of(row), int(row["fold"]))
        for row in rows if row.get("status") == "ok"
    }
    journal("confirmation_started", candidates=[row["run_id"] for row in candidates], seeds=seeds, folds=5)

    for seed in seeds:
        splitter = StratifiedKFold(5, shuffle=True, random_state=int(seed))
        for fold, (train_subject_i, test_subject_i) in enumerate(splitter.split(subjects, strata), 1):
            train_subjects, test_subjects = subjects[train_subject_i], subjects[test_subject_i]
            if np.intersect1d(train_subjects, test_subjects).size:
                raise RuntimeError("Patient leakage detected")
            train_idx = np.flatnonzero(np.isin(arrays["patient_id"], train_subjects))
            test_idx = np.flatnonzero(np.isin(arrays["patient_id"], test_subjects))
            for number, base in enumerate(candidates, 1):
                key = (base["run_id"], int(seed), fold)
                if key in completed:
                    continue
                config = {k: v for k, v in base.items() if not k.startswith("screening_") and k != "selection_role"}
                config.update({
                    "run_id": f"I200C_C{number}_seed{seed}_fold{fold}",
                    "stage": "INNOVATION200_CONFIRM",
                    "epochs": max(3, int(base.get("screening_epoch", 5))),
                    "fixed_epochs": True,
                })
                journal(
                    "fold_started", candidate_id=base["run_id"], run_id=config["run_id"],
                    seed=seed, fold=fold, train_subjects=len(train_subjects), test_subjects=len(test_subjects),
                )
                result = run_experiment(
                    config, arrays, train_idx, test_idx, OUT / "runs" / config["run_id"],
                    seed=int(seed) + fold, save_model=True,
                )
                result.update({
                    "candidate_id": base["run_id"],
                    "innovation_label": base["innovation_label"],
                    "selection_role": base["selection_role"],
                    "candidate_number": number,
                    "cv_seed": int(seed),
                    "training_seed": int(seed) + fold,
                    "fold": fold,
                    "status": "ok",
                    "train_subjects": len(train_subjects),
                    "test_subjects": len(test_subjects),
                    "patient_overlap": 0,
                    "checkpoint_saved": True,
                })
                rows.append(result)
                pd.DataFrame(rows).to_csv(results_path, index=False)
                journal("fold_completed", candidate_id=base["run_id"], seed=seed, fold=fold, result=result)

    frame = pd.DataFrame(rows)
    expected = len(candidates) * len(seeds) * 5
    if len(frame) != expected or frame.status.value_counts().to_dict() != {"ok": expected}:
        raise RuntimeError(f"Incomplete confirmation: {len(frame)}/{expected}")
    summary = frame.groupby(
        ["candidate_id", "innovation_label", "selection_role"], as_index=False
    ).agg(
        fits=("selection_score", "size"),
        mae_sbp_mean=("mae_sbp", "mean"), mae_sbp_sd=("mae_sbp", "std"),
        mae_dbp_mean=("mae_dbp", "mean"), mae_dbp_sd=("mae_dbp", "std"),
        tail_mae_sbp_mean=("tail_mae_sbp", "mean"), tail_mae_dbp_mean=("tail_mae_dbp", "mean"),
        high_bp_sensitivity_mean=("high_bp_sensitivity", "mean"),
        high_bp_specificity_mean=("high_bp_specificity", "mean"),
        normalized_score_mean=("selection_score", "mean"),
        clinical_score_mean=("clinical_selection_score", "mean"),
        parameters=("parameters", "first"), runtime_seconds=("runtime_seconds", "sum"),
    ).sort_values(["normalized_score_mean", "clinical_score_mean", "parameters"])
    summary.to_csv(OUT / "summary.csv", index=False)
    journal("confirmation_completed", rows=expected, summary=summary.to_dict("records"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
