from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TREND = ROOT / "phase2" / "external" / "datasets" / "uqvs" / "trend"
OUT = ROOT / "phase2" / "outputs" / "uqvs_forecast_gap"


def number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_case(path: Path) -> pd.DataFrame:
    # Alarm messages at the end of UQVS rows can contain unquoted commas.  The
    # first 19 fields are fixed and contain all variables needed here.
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header[:19] == [
            "Time", "RelativeTimeMilliseconds", "Clock", "HR", "ST-II", "Pulse",
            "SpO2", "Perf", "etCO2", "imCO2", "awRR", "NBP (Sys)", "NBP (Dia)",
            "NBP (Mean)", "NBP (Pulse)", "NBP (Time Remaining)", "ART (Sys)",
            "ART (Dia)", "ART (Mean)",
        ]
        for row in reader:
            if len(row) < 19:
                continue
            rows.append([number(row[i]) for i in (1, 3, 5, 6, 7, 11, 12, 13, 16, 17, 18)])
    frame = pd.DataFrame(
        rows,
        columns=["time_ms", "hr", "pulse", "spo2", "perf", "nbp_sbp", "nbp_dbp",
                 "nbp_map", "art_sbp", "art_dbp", "art_map"],
    )
    frame = frame[np.isfinite(frame.time_ms)].copy()
    frame["second"] = np.rint(frame.time_ms / 1000).astype(int)
    return frame.drop_duplicates("second", keep="last").set_index("second").sort_index()


def interval_mean(series: pd.Series, start: int, stop: int, min_fraction: float = 0.5) -> float:
    values = series.loc[(series.index >= start) & (series.index <= stop)].to_numpy(float)
    finite = values[np.isfinite(values)]
    expected = stop - start + 1
    if len(finite) < max(1, int(np.ceil(expected * min_fraction))):
        return np.nan
    return float(np.mean(finite))


def build_examples() -> pd.DataFrame:
    examples: list[dict] = []
    for path in sorted(TREND.glob("uq_vsd_case*_trenddata.csv")):
        case = path.stem.split("case", 1)[1].split("_", 1)[0]
        frame = load_case(path)
        if frame.empty:
            continue
        # Mirror Zhang et al. (IEEE SPL 2026): 60-s history, 30-s stride,
        # and target averaged from 4:40 to 5:20 after the history ends.
        for end in range(60, int(frame.index.max()) - 320 + 1, 30):
            current_sbp = interval_mean(frame.nbp_sbp, end - 40, end)
            current_dbp = interval_mean(frame.nbp_dbp, end - 40, end)
            future_sbp = interval_mean(frame.nbp_sbp, end + 280, end + 320)
            future_dbp = interval_mean(frame.nbp_dbp, end + 280, end + 320)
            if not np.all(np.isfinite([current_sbp, current_dbp, future_sbp, future_dbp])):
                continue
            examples.append(
                {
                    "case": case,
                    "history_end_second": end,
                    "current_sbp": current_sbp,
                    "current_dbp": current_dbp,
                    "future_sbp": future_sbp,
                    "future_dbp": future_dbp,
                    "delta_sbp": future_sbp - current_sbp,
                    "delta_dbp": future_dbp - current_dbp,
                }
            )
    result = pd.DataFrame(examples)
    if result.empty:
        raise RuntimeError("No valid UQVS forecast examples")
    result["major_transition"] = (
        (result.delta_sbp.abs() >= 10) | (result.delta_dbp.abs() >= 5)
    )
    result["direction_sbp"] = np.select(
        [result.delta_sbp <= -10, result.delta_sbp >= 10], [-1, 1], default=0
    )
    return result


def metric_rows(data: pd.DataFrame, pred_sbp: np.ndarray, pred_dbp: np.ndarray, name: str) -> list[dict]:
    rows = []
    masks = {
        "all": np.ones(len(data), dtype=bool),
        "stable": ~data.major_transition.to_numpy(bool),
        "major_transition": data.major_transition.to_numpy(bool),
    }
    for subset, mask in masks.items():
        if not mask.any():
            continue
        rows.append(
            {
                "model": name,
                "subset": subset,
                "n": int(mask.sum()),
                "mae_sbp": float(np.mean(np.abs(pred_sbp[mask] - data.future_sbp.to_numpy()[mask]))),
                "mae_dbp": float(np.mean(np.abs(pred_dbp[mask] - data.future_dbp.to_numpy()[mask]))),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_examples()
    data.to_csv(OUT / "FORECAST_EXAMPLES.csv", index=False)

    metrics: list[dict] = []
    metrics += metric_rows(
        data, data.current_sbp.to_numpy(), data.current_dbp.to_numpy(), "current_BP_persistence"
    )

    # A sample-random 80/20 split lets both training and test contain every
    # patient's identity.  This deliberately simple subject-mean predictor
    # quantifies how much a shuffled protocol can reward identity memorization.
    rng = np.random.default_rng(42)
    random_rows = []
    for repeat in range(100):
        test = rng.random(len(data)) < 0.2
        train = ~test
        global_sbp = float(data.loc[train, "future_sbp"].mean())
        global_dbp = float(data.loc[train, "future_dbp"].mean())
        means = data.loc[train].groupby("case")[["future_sbp", "future_dbp"]].mean()
        ps = np.array([means.future_sbp.get(case, global_sbp) for case in data.case])
        pdia = np.array([means.future_dbp.get(case, global_dbp) for case in data.case])
        for row in metric_rows(data.loc[test].reset_index(drop=True), ps[test], pdia[test], "random_split_subject_mean"):
            row["repeat"] = repeat
            random_rows.append(row)
    random_frame = pd.DataFrame(random_rows)
    random_frame.to_csv(OUT / "RANDOM_SPLIT_SUBJECT_MEMORY.csv", index=False)
    random_summary = (
        random_frame.groupby(["model", "subset"])[["n", "mae_sbp", "mae_dbp"]]
        .mean().reset_index()
    )
    metrics += random_summary.to_dict("records")

    # Leave-one-patient-out global mean: identity memorization is unavailable.
    lo_sbp = np.empty(len(data), dtype=float)
    lo_dbp = np.empty(len(data), dtype=float)
    for case in data.case.unique():
        test = data.case.eq(case).to_numpy()
        lo_sbp[test] = data.loc[~test, "future_sbp"].mean()
        lo_dbp[test] = data.loc[~test, "future_dbp"].mean()
    metrics += metric_rows(data, lo_sbp, lo_dbp, "patient_separated_global_mean")

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "BASELINE_METRICS.csv", index=False)

    persistence = metric_frame.query("model == 'current_BP_persistence' and subset == 'all'").iloc[0]
    transition = data.major_transition
    summary = {
        "cases_with_valid_pairs": int(data.case.nunique()),
        "examples": int(len(data)),
        "major_transition_definition": "abs(delta_sbp)>=10 OR abs(delta_dbp)>=5 mmHg",
        "major_transition_count": int(transition.sum()),
        "major_transition_prevalence": float(transition.mean()),
        "median_abs_delta_sbp": float(data.delta_sbp.abs().median()),
        "median_abs_delta_dbp": float(data.delta_dbp.abs().median()),
        "persistence_mae_sbp": float(persistence.mae_sbp),
        "persistence_mae_dbp": float(persistence.mae_dbp),
        "protocol_warning": (
            "The source IEEE SPL paper used 50%-overlapping windows and a shuffled 80/20 sample split. "
            "A patient-separated split and persistence skill were not reported."
        ),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()

