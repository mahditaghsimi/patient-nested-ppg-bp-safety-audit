from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import load_arrays, metrics
from phase2.analyze_icbme_medical_safety import classification_metrics, repeated_patient_crossfit_threshold
from phase2.run_confirmation_cv import subject_strata


LOCK = ROOT / "phase2/protocol/INNOVATION200_CONFIRMATION_LOCK.json"
OUT = ROOT / "phase2/outputs/innovation200_confirmation"


def patient_errors(patient: np.ndarray, y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.unique(patient)
    errors = np.asarray([
        np.abs(pred[patient == subject] - y[patient == subject]).mean(axis=0)
        for subject in subjects
    ])
    return subjects, errors


def paired_bootstrap(delta: np.ndarray, seed: int, iterations: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        sample = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[sample].mean(axis=1))
    values = np.concatenate(means)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    candidates = lock["finalists"]
    seeds = lock["confirmation"]["fresh_seeds"]
    fold_results = pd.read_csv(OUT / "fold_results.csv")
    expected = len(candidates) * len(seeds) * 5
    if len(fold_results) != expected or fold_results.status.value_counts().to_dict() != {"ok": expected}:
        raise RuntimeError("Confirmation is not complete")
    arrays = load_arrays()
    y = arrays["y"].astype(np.float32)
    patient = arrays["patient_id"].astype(str)
    global_std = y.std(axis=0)
    subjects, strata = subject_strata(arrays["patient_id"], arrays["y"])
    mean_oof_by_seed = []
    for seed in seeds:
        mean_oof = np.full_like(y, np.nan)
        splitter = StratifiedKFold(5, shuffle=True, random_state=int(seed))
        for train_subject_i, test_subject_i in splitter.split(subjects, strata):
            train_idx = np.flatnonzero(np.isin(arrays["patient_id"], subjects[train_subject_i]))
            test_idx = np.flatnonzero(np.isin(arrays["patient_id"], subjects[test_subject_i]))
            mean_oof[test_idx] = y[train_idx].mean(axis=0)
        if np.isnan(mean_oof).any():
            raise RuntimeError(f"Incomplete fold-mean baseline for seed {seed}")
        mean_oof_by_seed.append(mean_oof)
    fold_mean_prediction = np.mean(mean_oof_by_seed, axis=0).astype(np.float32)
    np.savez_compressed(
        OUT / "FOLD_MEAN_BASELINE.npz", y_true=y, patient_id=patient,
        prediction=fold_mean_prediction,
    )
    predictions: dict[str, np.ndarray] = {}
    summary_rows = []

    for number, candidate in enumerate(candidates, 1):
        seed_predictions = []
        for seed in seeds:
            oof = np.full_like(y, np.nan)
            seen = np.zeros(len(y), dtype=np.int8)
            for fold in range(1, 6):
                run_id = f"I200C_C{number}_seed{seed}_fold{fold}"
                with np.load(OUT / "runs" / run_id / "best_predictions.npz", allow_pickle=False) as data:
                    indices = data["indices"]
                    if np.any(seen[indices]):
                        raise RuntimeError(f"Duplicate OOF indices: {run_id}")
                    oof[indices] = data["y_pred"]
                    seen[indices] = 1
            if not np.all(seen == 1) or np.isnan(oof).any():
                raise RuntimeError(f"Incomplete OOF coverage for {candidate['run_id']} seed {seed}")
            seed_predictions.append(oof)
        averaged = np.mean(seed_predictions, axis=0).astype(np.float32)
        predictions[candidate["run_id"]] = averaged
        score = metrics(y, averaged, global_std)
        _, p_error = patient_errors(patient, y, averaged)
        summary_rows.append({
            "candidate_id": candidate["run_id"],
            "innovation_label": candidate["innovation_label"],
            "selection_role": candidate["selection_role"],
            **score,
            "patient_macro_mae_sbp": float(p_error[:, 0].mean()),
            "patient_macro_mae_dbp": float(p_error[:, 1].mean()),
            "parameters": int(fold_results[fold_results.candidate_id == candidate["run_id"]].parameters.iloc[0]),
        })

    new_ids = [row["run_id"] for row in candidates[1:]]
    synthetic = [
        {
            "run_id": "ENSEMBLE_ALL_NEW_EQUAL",
            "innovation_label": "equal_ensemble_of_four_new_finalists",
            "selection_role": "prespecified_equal_ensemble",
            "members": new_ids,
        },
        {
            "run_id": "ENSEMBLE_ALL_FIVE_EQUAL",
            "innovation_label": "equal_ensemble_prior_plus_four_new",
            "selection_role": "prespecified_equal_ensemble",
            "members": ["PRIOR_WINNER_FIXED", *new_ids],
        },
        {
            "run_id": "ENSEMBLE_BASELINE_ACCURACY_EQUAL",
            "innovation_label": "equal_ensemble_prior_plus_accuracy_finalist",
            "selection_role": "prespecified_equal_ensemble",
            "members": ["PRIOR_WINNER_FIXED", new_ids[0]],
        },
    ]
    for spec in synthetic:
        averaged = np.mean([predictions[member] for member in spec["members"]], axis=0).astype(np.float32)
        predictions[spec["run_id"]] = averaged
        score = metrics(y, averaged, global_std)
        _, p_error = patient_errors(patient, y, averaged)
        summary_rows.append({
            "candidate_id": spec["run_id"],
            "innovation_label": spec["innovation_label"],
            "selection_role": spec["selection_role"],
            **score,
            "patient_macro_mae_sbp": float(p_error[:, 0].mean()),
            "patient_macro_mae_dbp": float(p_error[:, 1].mean()),
            "parameters": int(sum(
                fold_results[fold_results.candidate_id == member].parameters.iloc[0]
                for member in spec["members"]
            )),
        })
    analysis_candidates = [*candidates, *synthetic]
    summary = pd.DataFrame(summary_rows).sort_values(["selection_score", "clinical_selection_score", "parameters"])
    summary.to_csv(OUT / "OOF_SUMMARY.csv", index=False)
    baseline_id = "PRIOR_WINNER_FIXED"
    _, baseline_error = patient_errors(patient, y, predictions[baseline_id])
    retained_prediction = predictions[baseline_id]
    high_truth = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    high_direct = (retained_prediction[:, 0] >= 140) | (retained_prediction[:, 1] >= 90)
    high_score = np.maximum(retained_prediction[:, 0] - 140, retained_prediction[:, 1] - 90)
    map_truth = (y[:, 0] + 2 * y[:, 1]) / 3
    map_prediction = (retained_prediction[:, 0] + 2 * retained_prediction[:, 1]) / 3
    low_truth = map_truth < 65
    low_direct = map_prediction < 65
    low_score = 65 - map_prediction
    event_safety = {
        "candidate_id": baseline_id,
        "high_bp": {
            "direct": classification_metrics(high_truth, high_direct),
            "auprc": float(average_precision_score(high_truth, high_score)),
            "auroc": float(roc_auc_score(high_truth, high_score)),
            "crossfit_90pct_specificity": repeated_patient_crossfit_threshold(
                patient, high_truth, high_score
            ),
        },
        "map_below_65": {
            "direct": classification_metrics(low_truth, low_direct),
            "auprc": float(average_precision_score(low_truth, low_score)),
            "auroc": float(roc_auc_score(low_truth, low_score)),
            "crossfit_90pct_specificity": repeated_patient_crossfit_threshold(
                patient, low_truth, low_score
            ),
        },
    }
    (OUT / "EVENT_SAFETY.json").write_text(
        json.dumps(event_safety, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _, mean_baseline_error = patient_errors(patient, y, fold_mean_prediction)
    fold_mean_comparison = {"metrics": metrics(y, fold_mean_prediction, global_std), "paired_prior_minus_mean": {}}
    for target_index, target in ((0, "SBP"), (1, "DBP"), (None, "MEAN")):
        left = baseline_error.mean(axis=1) if target_index is None else baseline_error[:, target_index]
        right = mean_baseline_error.mean(axis=1) if target_index is None else mean_baseline_error[:, target_index]
        delta = left - right
        ci_low, ci_high = paired_bootstrap(
            delta, seed=20260821 + (2 if target_index is None else target_index)
        )
        fold_mean_comparison["paired_prior_minus_mean"][target] = {
            "prior_patient_mae": float(left.mean()),
            "fold_mean_patient_mae": float(right.mean()),
            "delta": float(delta.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "wilcoxon_p": float(wilcoxon(left, right).pvalue),
        }
    (OUT / "FOLD_MEAN_BASELINE.json").write_text(
        json.dumps(fold_mean_comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paired_rows = []
    for candidate_index, candidate in enumerate(analysis_candidates[1:], 1):
        _, candidate_error = patient_errors(patient, y, predictions[candidate["run_id"]])
        for target_index, target in ((0, "SBP"), (1, "DBP"), (None, "MEAN")):
            left = candidate_error.mean(axis=1) if target_index is None else candidate_error[:, target_index]
            right = baseline_error.mean(axis=1) if target_index is None else baseline_error[:, target_index]
            delta = left - right
            ci_low, ci_high = paired_bootstrap(delta, seed=314159 + candidate_index * 10 + (2 if target_index is None else target_index))
            statistic, p_value = wilcoxon(left, right)
            paired_rows.append({
                "candidate_id": candidate["run_id"],
                "innovation_label": candidate["innovation_label"],
                "target": target,
                "candidate_patient_mae": float(left.mean()),
                "baseline_patient_mae": float(right.mean()),
                "delta_candidate_minus_baseline": float(delta.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "patients_improved_fraction": float(np.mean(delta < 0)),
                "wilcoxon_statistic": float(statistic),
                "p_unadjusted": float(p_value),
            })
    p_values = [row["p_unadjusted"] for row in paired_rows]
    for row, adjusted in zip(paired_rows, holm(p_values)):
        row["p_holm"] = adjusted
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "PAIRED_PATIENT_TESTS.csv", index=False)

    promoted = []
    for candidate in analysis_candidates[1:]:
        tests = paired[paired.candidate_id == candidate["run_id"]].set_index("target")
        if (
            tests.loc["MEAN", "ci_high"] < 0
            and tests.loc["MEAN", "p_holm"] < 0.05
            and tests.loc["SBP", "delta_candidate_minus_baseline"] < 0
            and tests.loc["DBP", "delta_candidate_minus_baseline"] < 0
        ):
            promoted.append(candidate["run_id"])
    winner_pool = summary[summary.candidate_id.isin(promoted)]
    if winner_pool.empty:
        winner_id = baseline_id
        status = "NO_CONFIRMED_IMPROVEMENT"
    else:
        winner_id = str(winner_pool.iloc[0].candidate_id)
        status = "CONFIRMED_IMPROVEMENT"
    winner_config = next(row for row in analysis_candidates if row["run_id"] == winner_id)
    original_by_id = {row["run_id"]: row for row in candidates}
    def transferable(row: dict) -> bool:
        if "members" in row:
            return all(original_by_id[member].get("engineered", "none") == "none" for member in row["members"])
        return row.get("engineered", "none") == "none"
    transferable_promoted = [
        row["run_id"] for row in analysis_candidates[1:]
        if row["run_id"] in promoted and transferable(row)
    ]
    if transferable_promoted:
        external_id = str(summary[summary.candidate_id.isin(transferable_promoted)].iloc[0].candidate_id)
    else:
        external_id = baseline_id
    external_config = next(row for row in analysis_candidates if row["run_id"] == external_id)
    winner_summary = summary[summary.candidate_id == winner_id].iloc[0].to_dict()
    baseline_summary = summary[summary.candidate_id == baseline_id].iloc[0].to_dict()
    np.savez_compressed(
        OUT / "OOF_PREDICTIONS.npz", y_true=y, patient_id=patient,
        **{f"pred_{index}": predictions[row["run_id"]] for index, row in enumerate(analysis_candidates)},
    )
    decision = {
        "status": status,
        "screening_methods": 200,
        "confirmation_fits": expected,
        "subjects": int(np.unique(patient).size),
        "segments": int(len(y)),
        "fresh_seeds": seeds,
        "folds_per_seed": 5,
        "promoted_candidate_ids": promoted,
        "winner_id": winner_id,
        "winner_config": winner_config,
        "winner_oof": winner_summary,
        "external_candidate_id": external_id,
        "external_candidate_config": external_config,
        "prior_winner_oof": baseline_summary,
        "promotion_rule": "paired patient mean error bootstrap CI<0, Holm p<0.05, and both target deltas<0",
        "claim_limit": "Selection is development-driven and repeated grouped OOF is not fully nested; external locked testing remains required.",
    }
    (OUT / "DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
