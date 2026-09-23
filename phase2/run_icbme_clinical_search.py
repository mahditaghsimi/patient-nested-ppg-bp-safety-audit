from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import load_arrays, run_experiment
from phase2.icbme_clinical_registry import all_configs, protocol


OUT = ROOT / "phase2/outputs/icbme_clinical_search"
LOCK = ROOT / "phase2/protocol/ICBME_CLINICAL_SEARCH_LOCK.json"


def journal(event: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=["all", "MEDOBJ", "ROBUST", "BALANCE", "REP", "OPT"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for this stage")
    if not LOCK.exists():
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(protocol(), ensure_ascii=False, indent=2), encoding="utf-8")
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    configs = locked["configs"]
    if args.stage != "all":
        configs = [config for config in configs if config["stage"] == args.stage]
    if args.limit is not None:
        configs = configs[: args.limit]
    arrays = load_arrays()
    train = np.flatnonzero(arrays["split"].astype(str) == "train")
    validation = np.flatnonzero(arrays["split"].astype(str) == "validation")
    if args.smoke:
        train, validation = train[:768], validation[:384]
        configs = configs[: min(8, len(configs))]
    results_path = OUT / ("smoke_results.csv" if args.smoke else "screening_results.csv")
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    completed = {row["run_id"] for row in rows if row.get("status", "ok") == "ok"}
    journal("search_started", smoke=args.smoke, stage=args.stage, config_count=len(configs), seed=args.seed)
    for ordinal, base in enumerate(configs, 1):
        if base["run_id"] in completed:
            continue
        config = dict(base)
        if args.smoke:
            config.update({"epochs": 1, "patience": 1, "batch_size": 128})
            config["run_id"] += "__smoke"
        print(f"[{ordinal}/{len(configs)}] {config['run_id']}", flush=True)
        journal("run_started", run_id=base["run_id"], config=config)
        try:
            result = run_experiment(config, arrays, train, validation, OUT / "runs" / config["run_id"], seed=args.seed)
            result.update({key: value for key, value in base.items() if key not in result})
            result["status"] = "ok"
        except Exception as exc:
            result = {"run_id": base["run_id"], "stage": base["stage"], "clinical_method": base["clinical_method"], "status": "failed", "error": repr(exc)}
            journal("run_failed", **result)
            rows.append(result)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            raise
        rows.append(result)
        pd.DataFrame(rows).to_csv(results_path, index=False)
        journal("run_completed", run_id=base["run_id"], result=result)
    frame = pd.DataFrame(rows)
    valid = frame[frame.status == "ok"].copy()
    if not valid.empty:
        valid = valid.sort_values(["clinical_selection_score", "mean_mae", "parameters"])
        valid.to_csv(OUT / ("smoke_ranking.csv" if args.smoke else "screening_ranking.csv"), index=False)
        print(valid[["run_id", "stage", "clinical_method", "mae_sbp", "mae_dbp", "tail_mae_sbp", "tail_mae_dbp", "high_bp_sensitivity", "clinical_selection_score"]].head(20).to_string(index=False))
    journal("search_completed", completed=int(len(valid)), failed=int((frame.status != "ok").sum()))


if __name__ == "__main__":
    main()
