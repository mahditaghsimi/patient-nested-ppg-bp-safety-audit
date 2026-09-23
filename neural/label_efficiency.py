from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, seed_everything
from neural.train_transfer import train_mode


def main() -> None:
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load("outputs/neural/cache/metadata.npz", allow_pickle=False) as meta, np.load(
        cfg["features"]["cache_path"], allow_pickle=False
    ) as features:
        arrays = {
            "split": meta["split"].copy(),
            "y": meta["y"].copy(),
            "patient": meta["patient_id"].copy(),
            "engineered": features["raw__fusion"].copy(),
        }
    output_dir = Path("outputs/neural/label_efficiency")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fraction in (0.10, 0.25, 0.50):
        for mode in ("scratch", "frozen", "finetuned"):
            run_name = f"{mode}_{int(100 * fraction):02d}pct"
            row = train_mode(
                mode,
                cfg,
                arrays,
                output_dir,
                device,
                train_fraction=fraction,
                epochs_override=int(cfg["neural"]["epochs_screen"]),
                run_name=run_name,
            )
            rows.append(row)
            pd.DataFrame(rows).sort_values(["train_fraction", "normalized_mae_score"]).to_csv(
                output_dir / "comparison.csv", index=False
            )
    full = pd.read_csv("outputs/neural/transfer_screen/comparison.csv")
    full = full[full["mode"].isin(["scratch", "frozen", "finetuned"])].copy()
    full["train_fraction"] = 1.0
    train_mask = arrays["split"].astype(str) == "train"
    full["train_patients"] = int(np.unique(arrays["patient"][train_mask]).size)
    full["train_samples"] = int(train_mask.sum())
    combined = pd.concat([pd.DataFrame(rows), full], ignore_index=True)
    combined.sort_values(["train_fraction", "normalized_mae_score"]).to_csv(
        output_dir / "comparison_with_100pct.csv", index=False
    )
    print(combined.sort_values(["train_fraction", "normalized_mae_score"]).to_string(index=False))


if __name__ == "__main__":
    main()
