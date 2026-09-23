from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.engine import load_arrays, run_experiment
from phase2.registry import INPUT_VARIANTS, OBJECTIVE_VARIANTS, stage_a, with_id, write_protocol


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2/outputs"
RUNS = OUT / "runs"
RESULTS = OUT / "search_results.csv"
JOURNAL = OUT / "execution_journal.jsonl"


def journal(event: str, **payload):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_many(configs: list[dict], arrays: dict, train_idx: np.ndarray, val_idx: np.ndarray, results: list[dict]):
    completed = {r["run_id"] for r in results}
    for config in configs:
        if config["run_id"] in completed:
            continue
        journal("run_started", run_id=config["run_id"], config=config)
        try:
            row = run_experiment(config, arrays, train_idx, val_idx, RUNS / config["run_id"])
            results.append(row)
            journal("run_completed", run_id=config["run_id"], result=row)
        except Exception as exc:
            row = {"run_id": config["run_id"], "stage": config["stage"], "status": "failed", "error": repr(exc)}
            results.append(row)
            journal("run_failed", run_id=config["run_id"], error=repr(exc), traceback=traceback.format_exc())
        pd.DataFrame(results).to_csv(RESULTS, index=False)


def ranked(results: list[dict], stage: str) -> pd.DataFrame:
    frame = pd.DataFrame([r for r in results if r.get("stage") == stage and "selection_score" in r])
    return frame.sort_values(["selection_score", "mean_mae", "worst_target_mae", "parameters"])


def main():
    protocol_path = ROOT / "phase2/protocol/PROTOCOL_LOCK.json"
    if not protocol_path.exists():
        write_protocol(protocol_path)
    arrays = load_arrays()
    train_idx = np.flatnonzero(arrays["split"] == "train")
    val_idx = np.flatnonzero(arrays["split"] == "validation")
    if np.intersect1d(np.unique(arrays["patient_id"][train_idx]), np.unique(arrays["patient_id"][val_idx])).size:
        raise RuntimeError("Patient leakage between screening train and validation")
    results = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
    journal("search_started", train_samples=len(train_idx), validation_samples=len(val_idx), test_samples_used=0)

    a = stage_a()
    run_many(a, arrays, train_idx, val_idx, results)
    a_rank = ranked(results, "A").drop_duplicates("backbone").head(4)
    b_configs = []
    count = 0
    for _, parent in a_rank.iterrows():
        base = next(c for c in a if c["run_id"] == parent.run_id)
        for variant in INPUT_VARIANTS:
            count += 1
            changes = {k: v for k, v in base.items() if k not in ("run_id", "stage")}
            changes.update(variant)
            changes["backbone"] = base["backbone"]
            b_configs.append(with_id("B", count, changes))
    (OUT / "stage_b_resolved_registry.json").write_text(json.dumps(b_configs, ensure_ascii=False, indent=2), encoding="utf-8")
    run_many(b_configs, arrays, train_idx, val_idx, results)

    b_rank = ranked(results, "B").head(4)
    c_configs = []
    count = 0
    for _, parent in b_rank.iterrows():
        base = next(c for c in b_configs if c["run_id"] == parent.run_id)
        for variant in OBJECTIVE_VARIANTS:
            count += 1
            changes = {k: v for k, v in base.items() if k not in ("run_id", "stage")}
            changes.update(variant)
            c_configs.append(with_id("C", count, changes))
    (OUT / "stage_c_resolved_registry.json").write_text(json.dumps(c_configs, ensure_ascii=False, indent=2), encoding="utf-8")
    run_many(c_configs, arrays, train_idx, val_idx, results)
    final = ranked(results, "C").head(3)
    final.to_csv(OUT / "selected_for_nested_cv.csv", index=False)
    journal("screening_completed", total_success=sum("selection_score" in r for r in results), finalists=final.run_id.tolist())
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()

