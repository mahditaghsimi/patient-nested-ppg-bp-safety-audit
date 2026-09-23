from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "phase2/outputs/innovation200_confirmation/fold_results.csv"


def main() -> None:
    frame = pd.read_csv(RESULTS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = RESULTS.with_name(f"fold_results_pre_dedup_{stamp}.csv")
    shutil.copy2(RESULTS, backup)

    extracted = frame.run_id.astype(str).str.extract(r"_seed(\d+)_fold(\d+)$")
    if extracted.isna().any().any():
        bad = frame.loc[extracted.isna().any(axis=1), "run_id"].tolist()
        raise RuntimeError(f"Unparseable run IDs: {bad[:5]}")
    frame["cv_seed"] = extracted[0].astype(int)
    frame["fold"] = extracted[1].astype(int)
    frame["training_seed"] = frame["cv_seed"] + frame["fold"]
    before = len(frame)
    frame = frame.drop_duplicates(["candidate_id", "cv_seed", "fold"], keep="last")
    frame = frame.sort_values(["cv_seed", "fold", "candidate_number"], kind="stable")
    frame.to_csv(RESULTS, index=False)
    print({
        "backup": str(backup),
        "rows_before": before,
        "unique_rows_after": len(frame),
        "duplicates_removed": before - len(frame),
        "unique_keys": int(frame[["candidate_id", "cv_seed", "fold"]].drop_duplicates().shape[0]),
    })


if __name__ == "__main__":
    main()
