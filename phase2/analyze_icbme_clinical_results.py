from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import metrics


CONF = ROOT / "phase2/outputs/icbme_clinical_confirmation"
OUT = ROOT / "phase2/outputs/icbme_clinical_analysis"
SEEDS = [42, 123, 2026]


def load_candidate_predictions(number: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    members, truth_reference = [], None
    for seed in SEEDS:
        prediction = np.full((n, 2), np.nan, np.float32)
        truth = np.full((n, 2), np.nan, np.float32)
        for fold in range(1, 6):
            path = CONF / "runs" / f"ICBME_C{number}_seed{seed}_fold{fold}" / "best_predictions.npz"
            with np.load(path, allow_pickle=False) as data:
                indices = data["indices"]
                prediction[indices] = data["y_pred"]
                truth[indices] = data["y_true"]
        if np.isnan(prediction).any() or np.isnan(truth).any():
            raise RuntimeError(f"Incomplete OOF predictions for candidate {number}, seed {seed}")
        if truth_reference is not None and not np.allclose(truth_reference, truth):
            raise RuntimeError("Ground truth differs across seeds")
        truth_reference = truth
        members.append(prediction)
    return truth_reference, np.asarray(members)


def patient_bootstrap(patient: np.ndarray, y: np.ndarray, pred: np.ndarray, iterations: int = 2000) -> dict:
    rng = np.random.default_rng(20260821)
    subjects = np.unique(patient)
    groups = [np.flatnonzero(patient == subject) for subject in subjects]
    values = np.empty((iterations, 4), np.float32)
    high = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    tail = (y[:, 0] < 95) | (y[:, 0] > 150) | (y[:, 1] < 55) | (y[:, 1] > 95)
    for iteration in range(iterations):
        chosen = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[i] for i in chosen])
        absolute = np.abs(pred[index] - y[index])
        values[iteration, :2] = absolute.mean(0)
        values[iteration, 2] = absolute[tail[index]].mean() if np.any(tail[index]) else np.nan
        predicted_high = (pred[index, 0] >= 140) | (pred[index, 1] >= 90)
        values[iteration, 3] = np.sum(predicted_high & high[index]) / max(np.sum(high[index]), 1)
    result = {}
    for column, name in enumerate(["mae_sbp", "mae_dbp", "tail_mean_mae", "high_bp_sensitivity"]):
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = np.nanquantile(values[:, column], [0.025, 0.975])
    return {key: float(value) for key, value in result.items()}


def selective_rows(label: str, y: np.ndarray, pred: np.ndarray, score: np.ndarray, descending: bool, source: str) -> list[dict]:
    order = np.argsort(score)
    if descending:
        order = order[::-1]
    rows = []
    for coverage in [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]:
        selected = order[: max(1, int(round(len(order) * coverage)))]
        absolute = np.abs(pred[selected] - y[selected])
        high = (y[selected, 0] >= 140) | (y[selected, 1] >= 90)
        predicted_high = (pred[selected, 0] >= 140) | (pred[selected, 1] >= 90)
        rows.append({
            "candidate": label, "selection_source": source, "coverage": coverage,
            "samples": len(selected), "mae_sbp": float(absolute[:, 0].mean()),
            "mae_dbp": float(absolute[:, 1].mean()),
            "high_bp_sensitivity": float(np.sum(high & predicted_high) / max(np.sum(high), 1)),
        })
    return rows


def cross_conformal(patient: np.ndarray, y: np.ndarray, pred: np.ndarray, alpha: float = 0.10) -> dict:
    subjects = np.unique(patient)
    assignment = {subject: i % 5 for i, subject in enumerate(subjects)}
    group = np.asarray([assignment[value] for value in patient])
    lower, upper = np.empty_like(pred), np.empty_like(pred)
    for fold in range(5):
        calibration = group != fold
        evaluate = group == fold
        residual = np.abs(y[calibration] - pred[calibration])
        quantile = np.quantile(residual, np.ceil((len(residual) + 1) * (1 - alpha)) / len(residual), axis=0, method="higher")
        lower[evaluate] = pred[evaluate] - quantile
        upper[evaluate] = pred[evaluate] + quantile
    covered = (y >= lower) & (y <= upper)
    return {
        "nominal_coverage": 1 - alpha,
        "sbp_coverage": float(covered[:, 0].mean()), "dbp_coverage": float(covered[:, 1].mean()),
        "sbp_mean_width": float((upper[:, 0] - lower[:, 0]).mean()),
        "dbp_mean_width": float((upper[:, 1] - lower[:, 1]).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads((CONF / "SELECTED_CANDIDATES.json").read_text(encoding="utf-8"))["candidates"]
    with np.load(ROOT / "outputs/neural/cache/metadata.npz", allow_pickle=False) as meta:
        y = meta["y"].copy()
        patient = meta["patient_id"].astype(str)
    with np.load(ROOT / "phase2/outputs/clinical_sqi/clinical_sqi.npz", allow_pickle=False) as quality_file:
        quality = quality_file["quality_score"].copy()
    train_std = y.std(0)
    summary_rows, selective, conformal_rows = [], [], []
    candidate_predictions, patient_errors = {}, {}
    for number, candidate in enumerate(selected, 1):
        truth, members = load_candidate_predictions(number, len(y))
        if not np.allclose(truth, y):
            raise RuntimeError("OOF target mismatch")
        pred = members.mean(0)
        label = candidate["clinical_method"]
        candidate_predictions[label] = pred
        row = {"candidate": label, "selection_role": candidate["selection_role"], **metrics(y, pred, train_std), **patient_bootstrap(patient, y, pred)}
        uncertainty = members.std(0).mean(1)
        absolute_mean = np.abs(pred - y).mean(1)
        rho, p_value = spearmanr(uncertainty, absolute_mean)
        row.update({"uncertainty_error_spearman": float(rho), "uncertainty_error_p": float(p_value)})
        summary_rows.append(row)
        selective += selective_rows(label, y, pred, quality, True, "waveform_SQI")
        selective += selective_rows(label, y, pred, uncertainty, False, "ensemble_uncertainty")
        conformal_rows.append({"candidate": label, **cross_conformal(patient, y, pred)})
        patient_errors[label] = np.asarray([np.abs(pred[patient == subject] - y[patient == subject]).mean(0) for subject in np.unique(patient)])

    baseline = next(candidate["clinical_method"] for candidate in selected if candidate["selection_role"] == "fixed_baseline")
    tests = []
    for candidate in candidate_predictions:
        if candidate == baseline:
            continue
        for target, column in [("SBP", 0), ("DBP", 1), ("MEAN", None)]:
            left = patient_errors[candidate] if column is None else patient_errors[candidate][:, column]
            right = patient_errors[baseline] if column is None else patient_errors[baseline][:, column]
            if column is None:
                left, right = left.mean(1), right.mean(1)
            statistic, p_value = wilcoxon(left, right)
            tests.append({
                "candidate": candidate, "baseline": baseline, "target": target,
                "mean_delta_candidate_minus_baseline": float(np.mean(left - right)),
                "wilcoxon_statistic": float(statistic), "p_unadjusted": float(p_value),
            })
    order = np.argsort([row["p_unadjusted"] for row in tests])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(tests) - rank) * tests[index]["p_unadjusted"])
        running = max(running, adjusted)
        tests[index]["p_holm"] = running

    pd.DataFrame(summary_rows).to_csv(OUT / "OOF_CLINICAL_SUMMARY.csv", index=False)
    pd.DataFrame(selective).to_csv(OUT / "SELECTIVE_PREDICTION.csv", index=False)
    pd.DataFrame(conformal_rows).to_csv(OUT / "CONFORMAL_90.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "PAIRED_PATIENT_TESTS.csv", index=False)
    np.savez_compressed(OUT / "OOF_PREDICTIONS.npz", y_true=y, patient_id=patient, quality=quality, **candidate_predictions)

    summary = pd.DataFrame(summary_rows).sort_values("clinical_selection_score")
    tests_frame = pd.DataFrame(tests)
    best = summary.iloc[0]
    baseline_row = summary[summary.candidate == baseline].iloc[0]
    improvement = float(((baseline_row.mae_sbp + baseline_row.mae_dbp) - (best.mae_sbp + best.mae_dbp)) / 2)
    confirmed = bool(best.candidate != baseline and improvement >= 0.10 and (tests_frame[(tests_frame.candidate == best.candidate) & (tests_frame.target == "MEAN")].p_holm < 0.05).all())
    decision = {
        "baseline": baseline,
        "best_clinical_score_candidate": str(best.candidate),
        "mean_mae_improvement_mmHg": improvement,
        "confirmed_primary_improvement": confirmed,
        "paper_policy": "update primary model claim only if confirmed_primary_improvement is true; otherwise retain existing model and report medical audit as secondary analysis",
    }
    (OUT / "DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(summary[["candidate", "mae_sbp", "mae_dbp", "tail_mae_sbp", "tail_mae_dbp", "high_bp_sensitivity", "high_bp_specificity", "clinical_selection_score", "uncertainty_error_spearman"]].to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
