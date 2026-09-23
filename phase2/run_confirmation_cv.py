from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.engine import load_arrays, run_experiment


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2/outputs/confirmation_cv"
SEARCH_OUT = ROOT / "phase2/outputs"
SEEDS = [42, 123, 2026]


def journal(event: str, **payload):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def subject_strata(patient: np.ndarray, y: np.ndarray):
    subjects = np.unique(patient)
    medians = np.asarray([np.median(y[patient == subject], axis=0) for subject in subjects])
    sbp = pd.qcut(medians[:, 0], 4, labels=False, duplicates="drop")
    dbp = pd.qcut(medians[:, 1], 3, labels=False, duplicates="drop")
    strata = np.asarray(sbp, int) * 3 + np.asarray(dbp, int)
    return subjects, strata


def load_finalist_configs():
    selected = pd.read_csv(SEARCH_OUT / "selected_for_nested_cv.csv")
    registry = json.loads((SEARCH_OUT / "stage_c_resolved_registry.json").read_text(encoding="utf-8"))
    by_id = {c["run_id"]: c for c in registry}
    configs = []
    for _, row in selected.iterrows():
        config = by_id[row.run_id].copy()
        config["epochs"] = int(max(3, row.epoch))
        config["fixed_epochs"] = True
        configs.append(config)
    return configs


def main():
    arrays = load_arrays()
    subjects, strata = subject_strata(arrays["patient_id"], arrays["y"])
    configs = load_finalist_configs()
    rows = pd.read_csv(OUT / "fold_results.csv").to_dict("records") if (OUT / "fold_results.csv").exists() else []
    completed = {(r["candidate_id"], int(r["seed"]), int(r["fold"])) for r in rows if r.get("status", "ok") == "ok"}
    for seed in SEEDS:
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (train_subject_i, test_subject_i) in enumerate(folds.split(subjects, strata), 1):
            train_subjects, test_subjects = subjects[train_subject_i], subjects[test_subject_i]
            train_idx = np.flatnonzero(np.isin(arrays["patient_id"], train_subjects))
            test_idx = np.flatnonzero(np.isin(arrays["patient_id"], test_subjects))
            assert not np.intersect1d(train_subjects, test_subjects).size
            for candidate_number, base in enumerate(configs, 1):
                key = (base["run_id"], seed, fold)
                if key in completed:
                    continue
                config = base.copy()
                config["stage"] = "CV"
                config["run_id"] = f"CV_c{candidate_number}_seed{seed}_fold{fold}"
                journal("fold_started", candidate_id=base["run_id"], run_id=config["run_id"], seed=seed, fold=fold, train_subjects=len(train_subjects), test_subjects=len(test_subjects))
                result = run_experiment(config, arrays, train_idx, test_idx, OUT / "runs" / config["run_id"], seed=seed)
                result.update({"candidate_id": base["run_id"], "candidate_number": candidate_number, "fold": fold})
                rows.append(result)
                pd.DataFrame(rows).to_csv(OUT / "fold_results.csv", index=False)
                journal("fold_completed", candidate_id=base["run_id"], run_id=config["run_id"], seed=seed, fold=fold, metrics=result)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("candidate_id").agg(
        folds=("selection_score", "size"),
        selection_score_mean=("selection_score", "mean"), selection_score_sd=("selection_score", "std"),
        mae_sbp_mean=("mae_sbp", "mean"), mae_sbp_sd=("mae_sbp", "std"),
        mae_dbp_mean=("mae_dbp", "mean"), mae_dbp_sd=("mae_dbp", "std"),
        pearson_sbp_mean=("pearson_sbp", "mean"), pearson_dbp_mean=("pearson_dbp", "mean"),
    ).sort_values("selection_score_mean")
    summary.to_csv(OUT / "summary.csv")
    winner = summary.index[0]
    winner_config = next(c for c in configs if c["run_id"] == winner)
    (OUT / "winner.json").write_text(json.dumps({"candidate_id": winner, "config": winner_config, "summary": summary.loc[winner].to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
    journal("confirmation_completed", winner=winner, summary=summary.reset_index().to_dict("records"))
    print(summary.to_string())


if __name__ == "__main__":
    main()

