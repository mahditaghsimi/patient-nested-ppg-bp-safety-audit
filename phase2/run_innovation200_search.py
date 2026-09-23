from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import load_arrays, run_experiment


OUT = ROOT / "phase2/outputs/innovation200_search"
LOCK = ROOT / "phase2/protocol/INNOVATION200_SEARCH_LOCK.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def journal(event: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=["all", "ARCH", "INPUT", "OBJECTIVE", "TRAIN"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if not LOCK.exists():
        raise FileNotFoundError(f"Create the immutable protocol lock first: {LOCK}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for this stage")
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    assert locked["locked_before_training"] is True
    assert locked["method_count"] == 200
    configs = locked["configs"]
    if args.stage != "all":
        configs = [row for row in configs if row["stage"] == args.stage]
    if args.limit is not None:
        configs = configs[: args.limit]

    arrays = load_arrays()
    train = np.flatnonzero(arrays["split"].astype(str) == "train")
    validation = np.flatnonzero(arrays["split"].astype(str) == "validation")
    results_name = "smoke_results.csv" if args.smoke else "screening_results.csv"
    ranking_name = "smoke_ranking.csv" if args.smoke else "screening_ranking.csv"
    if args.smoke:
        train, validation = train[:1024], validation[:512]
        # One representative from every newly implemented backbone.
        wanted = ["adaptive_resnet", "morphology_resnet", "longconv", "conv_transformer_lite"]
        configs = [next(row for row in configs if row["backbone"] == name) for name in wanted]

    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / results_name
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    completed = {str(row["run_id"]).removesuffix("__smoke") for row in rows if row.get("status") == "ok"}
    metadata = {
        "protocol_sha256": sha256(LOCK),
        "seed": args.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "train_subjects": int(np.unique(arrays["patient_id"][train]).size),
        "validation_subjects": int(np.unique(arrays["patient_id"][validation]).size),
        "train_segments": int(len(train)),
        "validation_segments": int(len(validation)),
        "smoke": args.smoke,
    }
    (OUT / ("smoke_environment.json" if args.smoke else "environment.json")).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    journal("search_started", stage=args.stage, count=len(configs), **metadata)

    for ordinal, base in enumerate(configs, 1):
        if base["run_id"] in completed:
            continue
        config = dict(base)
        if args.smoke:
            config.update({"epochs": 1, "patience": 1, "batch_size": 128})
            config["run_id"] += "__smoke"
        print(f"[{ordinal}/{len(configs)}] {config['run_id']} {config['innovation_label']}", flush=True)
        journal("run_started", ordinal=ordinal, run_id=config["run_id"], config=config)
        try:
            result = run_experiment(
                config, arrays, train, validation, OUT / "runs" / config["run_id"],
                seed=args.seed, save_model=False,
            )
            result.update({key: value for key, value in base.items() if key not in result})
            result["status"] = "ok"
            journal("run_completed", ordinal=ordinal, run_id=config["run_id"], result=result)
        except Exception as exc:
            result = {
                "run_id": config["run_id"], "stage": base["stage"],
                "innovation_label": base["innovation_label"], "status": "failed", "error": repr(exc),
            }
            journal("run_failed", ordinal=ordinal, **result)
            print(f"FAILED {config['run_id']}: {exc!r}", file=sys.stderr, flush=True)
        rows.append(result)
        pd.DataFrame(rows).to_csv(results_path, index=False)

    frame = pd.DataFrame(rows)
    valid = frame[frame["status"] == "ok"].copy()
    if not valid.empty:
        valid = valid.sort_values(
            ["selection_score", "clinical_selection_score", "mean_mae", "parameters"],
            kind="stable",
        )
        valid.to_csv(OUT / ranking_name, index=False)
        columns = [
            "run_id", "stage", "innovation_label", "mae_sbp", "mae_dbp",
            "tail_mae_sbp", "tail_mae_dbp", "high_bp_sensitivity",
            "selection_score", "runtime_seconds", "parameters",
        ]
        print(valid[columns].head(25).to_string(index=False))
    failed = int((frame["status"] != "ok").sum())
    journal("search_completed", completed=int(len(valid)), failed=failed)
    print(json.dumps({"completed": int(len(valid)), "failed": failed, "results": str(results_path)}, indent=2))
    if failed and not args.smoke:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
