from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import load_arrays, run_experiment


OUT = ROOT / "phase2/outputs/innovation200_search/reference_prior_winner"
CONFIG = {
    "run_id": "I200_REFERENCE_PRIOR_WINNER_seed314159",
    "stage": "REFERENCE",
    "backbone": "resnet",
    "view": "multiview",
    "duration_seconds": 10.0,
    "engineered": "none",
    "head": "multitask",
    "loss": "multitask",
    "augmentation": "jitter_scale",
    "balance": "none",
    "dropout": 0.15,
    "head_hidden": 192,
    "epochs": 5,
    "patience": 2,
    "batch_size": 192,
    "learning_rate": 1e-3,
    "weight_decay": 2e-4,
    "optimizer": "adamw",
    "scheduler": "none",
}


def main() -> None:
    locked = ROOT / "phase2/protocol/INNOVATION200_CONFIRMATION_LOCK.json"
    reference_path = OUT / "REFERENCE_RESULT.json"
    if locked.exists() and reference_path.exists():
        lock = json.loads(locked.read_text(encoding="utf-8"))
        current_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if current_hash == lock.get("reference_result_sha256"):
            print(reference_path.read_text(encoding="utf-8"))
            return
        raise RuntimeError(
            "Confirmation is already locked and the current reference artifact does not match its hash; "
            "refusing to overwrite audit evidence."
        )
    arrays = load_arrays()
    train = np.flatnonzero(arrays["split"].astype(str) == "train")
    validation = np.flatnonzero(arrays["split"].astype(str) == "validation")
    result = run_experiment(CONFIG, arrays, train, validation, OUT, seed=314159, save_model=False)
    result["role"] = "same-seed reference; not counted among 200 new methods"
    (OUT / "REFERENCE_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
