from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "phase2/outputs/confirmation_cv"
OUT = ROOT / "phase2/outputs/analysis"
SEEDS = [42, 123, 2026]


def metrics(y, pred):
    err = pred - y
    return {
        "mae_sbp": float(np.abs(err[:, 0]).mean()),
        "mae_dbp": float(np.abs(err[:, 1]).mean()),
        "rmse_sbp": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_dbp": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "bias_sbp": float(err[:, 0].mean()),
        "bias_dbp": float(err[:, 1].mean()),
        "pearson_sbp": float(np.corrcoef(y[:, 0], pred[:, 0])[0, 1]),
        "pearson_dbp": float(np.corrcoef(y[:, 1], pred[:, 1])[0, 1]),
    }


def load_oof(candidate_number, seed, n):
    pred = np.full((n, 2), np.nan, np.float32)
    true = np.full((n, 2), np.nan, np.float32)
    for fold in range(1, 6):
        path = CV / "runs" / f"CV_c{candidate_number}_seed{seed}_fold{fold}" / "best_predictions.npz"
        with np.load(path) as data:
            pred[data["indices"]] = data["y_pred"]
            true[data["indices"]] = data["y_true"]
    if np.isnan(pred).any() or np.isnan(true).any():
        raise RuntimeError(f"Incomplete OOF predictions for candidate {candidate_number}, seed {seed}")
    return true, pred


def bootstrap(patient, y, pred, iterations=2000, seed=20260820):
    rng = np.random.default_rng(seed)
    subjects = np.unique(patient)
    subject_indices = [np.flatnonzero(patient == p) for p in subjects]
    values = np.empty((iterations, 2), np.float32)
    for b in range(iterations):
        selected = rng.integers(0, len(subjects), len(subjects))
        index = np.concatenate([subject_indices[i] for i in selected])
        values[b] = np.abs(pred[index] - y[index]).mean(0)
    return {
        "iterations": iterations,
        "mae_sbp_ci95": np.quantile(values[:, 0], [0.025, 0.975]).tolist(),
        "mae_dbp_ci95": np.quantile(values[:, 1], [0.025, 0.975]).tolist(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "outputs/neural/cache/metadata.npz") as meta:
        patient = meta["patient_id"].copy()
        y = meta["y"].copy()
    selected = pd.read_csv(ROOT / "phase2/outputs/selected_for_nested_cv.csv")
    labels = {i + 1: row.run_id for i, row in selected.iterrows()}
    predictions = {}
    seed_rows = []
    for number, candidate_id in labels.items():
        seed_predictions = []
        for seed in SEEDS:
            true, pred = load_oof(number, seed, len(y))
            if not np.allclose(true, y):
                raise RuntimeError("OOF ground truth is misaligned")
            seed_predictions.append(pred)
            seed_rows.append({"candidate_id": candidate_id, "seed": seed, **metrics(y, pred)})
        predictions[candidate_id] = np.mean(seed_predictions, axis=0)
    rows = []
    for candidate_id, pred in predictions.items():
        rows.append({"model": candidate_id, "kind": "candidate_seed_ensemble", **metrics(y, pred), **bootstrap(patient, y, pred)})
    simple = np.mean(list(predictions.values()), axis=0)
    rows.append({"model": "simple_mean_all_3", "kind": "ensemble_preregistered_equal_weight", **metrics(y, simple), **bootstrap(patient, y, simple)})
    candidate_ids = list(predictions)
    grid = []
    for a in np.linspace(0, 1, 11):
        for b in np.linspace(0, 1 - a, int(round((1 - a) * 10)) + 1):
            c = 1 - a - b
            pred = a * predictions[candidate_ids[0]] + b * predictions[candidate_ids[1]] + c * predictions[candidate_ids[2]]
            m = metrics(y, pred)
            grid.append({"w1": a, "w2": b, "w3": c, "mean_mae": (m["mae_sbp"] + m["mae_dbp"]) / 2, **m})
    pd.DataFrame(grid).sort_values("mean_mae").to_csv(OUT / "exploratory_ensemble_grid.csv", index=False)
    subject_errors = {}
    for name, pred in {**predictions, "simple_mean_all_3": simple}.items():
        subject_errors[name] = np.asarray([np.abs(pred[patient == p] - y[patient == p]).mean(0) for p in np.unique(patient)])
    paired = []
    names = list(subject_errors)
    for left, right in itertools.combinations(names, 2):
        for target, column in (("SBP", 0), ("DBP", 1)):
            delta = subject_errors[left][:, column] - subject_errors[right][:, column]
            stat, p = wilcoxon(delta, alternative="two-sided")
            paired.append({"left": left, "right": right, "target": target, "mean_mae_delta_left_minus_right": float(delta.mean()), "wilcoxon_statistic": float(stat), "p_value_unadjusted": float(p)})
    # Holm family-wise error correction across all paired target-wise tests.
    order = np.argsort([row["p_value_unadjusted"] for row in paired])
    adjusted = np.empty(len(paired), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(paired) - rank) * paired[index]["p_value_unadjusted"])
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(paired, adjusted):
        row["p_value_holm"] = float(value)
    pd.DataFrame(seed_rows).to_csv(OUT / "seed_level_oof_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "oof_ensemble_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_patient_tests.csv", index=False)
    np.savez_compressed(OUT / "oof_predictions.npz", y_true=y, patient_id=patient, simple_ensemble=simple, **{f"pred_{i+1}": predictions[c] for i, c in enumerate(candidate_ids)})
    print(pd.DataFrame(rows).drop(columns=["mae_sbp_ci95", "mae_dbp_ci95"]).to_string(index=False))


if __name__ == "__main__":
    main()
