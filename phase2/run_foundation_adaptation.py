from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.engine import load_arrays, run_experiment
from phase2.registry import with_id


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2/outputs/foundation_adaptation"


def configs():
    protocol = json.loads((ROOT / "phase2/protocol/PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    base = next(c for c in protocol["stage_A"]["configs"] if c["backbone"] == "papagei" and c.get("engineered") == "none" and c.get("foundation_mode") == "frozen")
    variants = [
        {"foundation_mode": "last_blocks", "learning_rate": 1e-4, "head": "direct", "loss": "huber"},
        {"foundation_mode": "full", "learning_rate": 5e-5, "head": "direct", "loss": "huber"},
        {"foundation_mode": "last_blocks", "learning_rate": 1e-4, "head": "multitask", "loss": "multitask", "augmentation": "jitter_scale"},
        {"foundation_mode": "full", "learning_rate": 5e-5, "head": "multitask", "loss": "multitask", "augmentation": "jitter_scale"},
        {"foundation_mode": "last_blocks", "learning_rate": 1e-4, "head": "direct", "loss": "l1"},
    ]
    output = []
    for i, variant in enumerate(variants, 1):
        changes = {k: v for k, v in base.items() if k not in ("run_id", "stage")}
        changes.update(variant)
        changes.update({"epochs": 10, "batch_size": 128, "patience": 4})
        output.append(with_id("F", i, changes))
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays()
    train_idx = np.flatnonzero(arrays["split"] == "train")
    val_idx = np.flatnonzero(arrays["split"] == "validation")
    rows = []
    for config in configs():
        rows.append(run_experiment(config, arrays, train_idx, val_idx, OUT / "runs" / config["run_id"]))
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    frame = pd.DataFrame(rows).sort_values("selection_score")
    frame.to_csv(OUT / "results.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
