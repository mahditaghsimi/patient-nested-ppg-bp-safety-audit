#!/usr/bin/env python3
"""Five-fold patient-nested evaluation of the locked 200-method selection procedure."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.engine import load_arrays, metrics, run_experiment
from phase2.run_confirmation_cv import subject_strata
from phase2.analyze_icbme_medical_safety import classification_metrics, repeated_patient_crossfit_threshold


OUT = ROOT / "phase2/outputs/nested200"
LOCK = ROOT / "phase2/protocol/INNOVATION200_SEARCH_LOCK.json"
OUTER_SEED = 90421
INNER_SEED_BASE = 271828


PRIOR = {
    "run_id": "PRIOR_WINNER_FIXED", "stage": "NESTED_REFERENCE",
    "innovation_label": "historical_resnet_multiview10_multitask",
    "backbone": "resnet", "view": "multiview", "duration_seconds": 10.0,
    "engineered": "none", "head": "multitask", "loss": "multitask",
    "augmentation": "jitter_scale", "balance": "none", "dropout": 0.15,
    "head_hidden": 192, "epochs": 5, "patience": 2, "batch_size": 192,
    "learning_rate": 1e-3, "weight_decay": 2e-4, "optimizer": "adamw", "scheduler": "none",
}


def journal(event: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}, ensure_ascii=False) + "\n")


def macro(patient: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> dict:
    subjects = np.unique(patient)
    errors = np.asarray([np.abs(prediction[patient == subject] - y[patient == subject]).mean(0) for subject in subjects])
    return {"patient_macro_mae_sbp": float(errors[:, 0].mean()), "patient_macro_mae_dbp": float(errors[:, 1].mean())}


def full_metrics(patient: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> dict:
    value = metrics(y, prediction, y.std(0))
    error = prediction - y
    for column, target in enumerate(("sbp", "dbp")):
        sd = float(error[:, column].std(ddof=1))
        value[f"error_sd_{target}"] = sd
        value[f"bland_altman_lower_{target}"] = float(error[:, column].mean() - 1.96 * sd)
        value[f"bland_altman_upper_{target}"] = float(error[:, column].mean() + 1.96 * sd)
    value.update(macro(patient, y, prediction))
    return value


def paired_patient_test(patient: np.ndarray, y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> dict:
    subjects = np.unique(patient)
    delta = np.asarray([
        np.abs(candidate[patient == subject] - y[patient == subject]).mean()
        - np.abs(baseline[patient == subject] - y[patient == subject]).mean()
        for subject in subjects
    ])
    statistic, p = wilcoxon(delta)
    rng = np.random.default_rng(99017)
    bootstrap = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(20000)])
    return {
        "patient_mean_mae_delta_candidate_minus_baseline": float(delta.mean()),
        "ci_low": float(np.quantile(bootstrap, .025)), "ci_high": float(np.quantile(bootstrap, .975)),
        "wilcoxon_statistic": float(statistic), "wilcoxon_p": float(p), "patients": len(subjects),
    }


def event_safety(patient: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> dict:
    high_truth = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    high_direct = (prediction[:, 0] >= 140) | (prediction[:, 1] >= 90)
    high_score = np.maximum(prediction[:, 0] - 140, prediction[:, 1] - 90)
    map_truth = (y[:, 0] + 2 * y[:, 1]) / 3
    map_prediction = (prediction[:, 0] + 2 * prediction[:, 1]) / 3
    low_truth = map_truth < 65
    low_direct = map_prediction < 65
    low_score = 65 - map_prediction
    return {
        "candidate": "nested_selected_procedure",
        "high_bp": {
            "definition": "operational: SBP>=140 or DBP>=90 mmHg",
            "direct": classification_metrics(high_truth, high_direct),
            "auprc": float(average_precision_score(high_truth, high_score)),
            "auroc": float(roc_auc_score(high_truth, high_score)),
            "crossfit_90pct_specificity": repeated_patient_crossfit_threshold(patient, high_truth, high_score),
        },
        "map_below_65": {
            "definition": "derived MAP approximation (SBP+2*DBP)/3 < 65 mmHg",
            "direct": classification_metrics(low_truth, low_direct),
            "auprc": float(average_precision_score(low_truth, low_score)),
            "auroc": float(roc_auc_score(low_truth, low_score)),
            "crossfit_90pct_specificity": repeated_patient_crossfit_threshold(patient, low_truth, low_score),
        },
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the nested-200 run")
    protocol = json.loads(LOCK.read_text(encoding="utf-8"))
    configs = [dict(row) for row in protocol["configs"]] + [dict(PRIOR)]
    if len(protocol["configs"]) != 200 or len({row["run_id"] for row in protocol["configs"]}) != 200:
        raise RuntimeError("Locked registry is not exactly 200 unique methods")
    arrays = load_arrays()
    subjects, strata = subject_strata(arrays["patient_id"], arrays["y"])
    outer = list(StratifiedKFold(5, shuffle=True, random_state=OUTER_SEED).split(subjects, strata))
    OUT.mkdir(parents=True, exist_ok=True)
    inner_results_path = OUT / "INNER_SCREEN_RESULTS.csv"
    inner_rows = pd.read_csv(inner_results_path).to_dict("records") if inner_results_path.exists() else []
    completed_inner = {(int(row["outer_fold"]), str(row["candidate_id"])) for row in inner_rows if row.get("status") == "ok"}
    outer_results_path = OUT / "OUTER_FOLD_RESULTS.csv"
    outer_rows = pd.read_csv(outer_results_path).to_dict("records") if outer_results_path.exists() else []
    completed_outer = {int(row["outer_fold"]) for row in outer_rows if row.get("status") == "ok"}
    selected_rows = []

    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer, 1):
        outer_train_subjects, outer_test_subjects = subjects[outer_train_i], subjects[outer_test_i]
        local_strata = strata[outer_train_i]
        split = StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=INNER_SEED_BASE + outer_fold)
        inner_train_local, inner_validation_local = next(split.split(outer_train_subjects, local_strata))
        inner_train_subjects = outer_train_subjects[inner_train_local]
        inner_validation_subjects = outer_train_subjects[inner_validation_local]
        if np.intersect1d(inner_train_subjects, inner_validation_subjects).size or np.intersect1d(outer_train_subjects, outer_test_subjects).size:
            raise RuntimeError("Nested patient overlap")
        if np.intersect1d(np.concatenate([inner_train_subjects, inner_validation_subjects]), outer_test_subjects).size:
            raise RuntimeError("Outer-test patient entered inner selection")
        inner_train_idx = np.flatnonzero(np.isin(arrays["patient_id"], inner_train_subjects))
        inner_validation_idx = np.flatnonzero(np.isin(arrays["patient_id"], inner_validation_subjects))
        outer_train_idx = np.flatnonzero(np.isin(arrays["patient_id"], outer_train_subjects))
        outer_test_idx = np.flatnonzero(np.isin(arrays["patient_id"], outer_test_subjects))
        split_record = {
            "outer_fold": outer_fold, "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED_BASE + outer_fold,
            "inner_train_subjects": inner_train_subjects.tolist(), "inner_validation_subjects": inner_validation_subjects.tolist(),
            "outer_test_subjects": outer_test_subjects.tolist(),
        }
        split_path = OUT / "splits" / f"outer_fold_{outer_fold}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split_record, indent=2) + "\n", encoding="utf-8")
        journal("outer_fold_started", outer_fold=outer_fold, inner_train=len(inner_train_subjects), inner_validation=len(inner_validation_subjects), outer_test=len(outer_test_subjects))

        for ordinal, base in enumerate(configs, 1):
            key = (outer_fold, base["run_id"])
            if key in completed_inner:
                continue
            config = dict(base)
            original_id = config["run_id"]
            config["run_id"] = f"N200_O{outer_fold}_{original_id}"
            config["stage"] = "NESTED_INNER_SCREEN"
            result = run_experiment(
                config, arrays, inner_train_idx, inner_validation_idx,
                OUT / "inner_runs" / f"outer_{outer_fold}" / original_id,
                seed=INNER_SEED_BASE + outer_fold, save_model=False,
            )
            result.update({"outer_fold": outer_fold, "candidate_id": original_id, "status": "ok", "ordinal": ordinal})
            inner_rows.append(result)
            pd.DataFrame(inner_rows).to_csv(inner_results_path, index=False)
            journal("inner_candidate_completed", outer_fold=outer_fold, ordinal=ordinal, candidate_id=original_id, selection_score=result["selection_score"])

        fold_frame = pd.DataFrame(inner_rows)
        fold_frame = fold_frame[(fold_frame.outer_fold == outer_fold) & (fold_frame.status == "ok")]
        if len(fold_frame) != 201:
            raise RuntimeError(f"Outer fold {outer_fold}: expected 201 completed inner candidates, found {len(fold_frame)}")
        winner = fold_frame.sort_values(["selection_score", "clinical_selection_score", "mean_mae", "parameters"], kind="stable").iloc[0]
        base = next(row for row in configs if row["run_id"] == winner.candidate_id)
        selected_rows.append({
            "outer_fold": outer_fold, "candidate_id": winner.candidate_id, "inner_epoch": int(winner.epoch),
            "inner_mae_sbp": float(winner.mae_sbp), "inner_mae_dbp": float(winner.mae_dbp),
            "inner_selection_score": float(winner.selection_score),
        })
        pd.DataFrame(selected_rows).to_csv(OUT / "OUTER_SELECTIONS.csv", index=False)
        if outer_fold in completed_outer:
            continue

        final_config = dict(base)
        final_config.update({
            "run_id": f"N200_OUTER_SELECTED_fold{outer_fold}", "stage": "NESTED_OUTER_EVAL",
            "epochs": max(1, int(winner.epoch)), "fixed_epochs": True,
        })
        result = run_experiment(
            final_config, arrays, outer_train_idx, outer_test_idx,
            OUT / "outer_runs" / f"selected_fold{outer_fold}", seed=OUTER_SEED + outer_fold, save_model=True,
        )
        prior_config = dict(PRIOR)
        prior_config.update({"run_id": f"N200_OUTER_PRIOR_fold{outer_fold}", "stage": "NESTED_OUTER_REFERENCE", "fixed_epochs": True})
        prior_result = run_experiment(
            prior_config, arrays, outer_train_idx, outer_test_idx,
            OUT / "outer_runs" / f"prior_fold{outer_fold}", seed=OUTER_SEED + outer_fold, save_model=True,
        )
        result.update({
            "outer_fold": outer_fold, "selected_candidate_id": winner.candidate_id, "inner_selected_epoch": int(winner.epoch),
            "status": "ok", "train_subjects": len(outer_train_subjects), "test_subjects": len(outer_test_subjects),
            "patient_overlap": 0, "prior_mae_sbp": prior_result["mae_sbp"], "prior_mae_dbp": prior_result["mae_dbp"],
        })
        outer_rows.append(result)
        pd.DataFrame(outer_rows).to_csv(outer_results_path, index=False)
        journal("outer_fold_completed", outer_fold=outer_fold, selection=selected_rows[-1], result=result)

    if len(outer_rows) != 5:
        raise RuntimeError(f"Expected five outer folds, found {len(outer_rows)}")
    selected_oof = np.full_like(arrays["y"], np.nan, dtype=np.float32)
    prior_oof = np.full_like(arrays["y"], np.nan, dtype=np.float32)
    fold_mean_oof = np.full_like(arrays["y"], np.nan, dtype=np.float32)
    coverage = np.zeros(len(arrays["y"]), np.int8)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer, 1):
        outer_train_subjects, outer_test_subjects = subjects[outer_train_i], subjects[outer_test_i]
        train_idx = np.flatnonzero(np.isin(arrays["patient_id"], outer_train_subjects))
        test_idx = np.flatnonzero(np.isin(arrays["patient_id"], outer_test_subjects))
        with np.load(OUT / "outer_runs" / f"selected_fold{outer_fold}" / "best_predictions.npz", allow_pickle=False) as data:
            if not np.array_equal(np.sort(data["indices"]), np.sort(test_idx)):
                raise RuntimeError("Selected outer prediction index mismatch")
            selected_oof[data["indices"]] = data["y_pred"]
        with np.load(OUT / "outer_runs" / f"prior_fold{outer_fold}" / "best_predictions.npz", allow_pickle=False) as data:
            if not np.array_equal(np.sort(data["indices"]), np.sort(test_idx)):
                raise RuntimeError("Prior outer prediction index mismatch")
            prior_oof[data["indices"]] = data["y_pred"]
        fold_mean_oof[test_idx] = arrays["y"][train_idx].mean(0)
        coverage[test_idx] += 1
    if not np.all(coverage == 1) or np.isnan(selected_oof).any() or np.isnan(prior_oof).any():
        raise RuntimeError("Nested OOF coverage failure")
    summary = {
        "nested_selected_procedure": full_metrics(arrays["patient_id"], arrays["y"], selected_oof),
        "historical_fixed_prior": full_metrics(arrays["patient_id"], arrays["y"], prior_oof),
        "outer_fold_training_mean": full_metrics(arrays["patient_id"], arrays["y"], fold_mean_oof),
    }
    tests = {
        "selected_vs_fold_mean": paired_patient_test(arrays["patient_id"], arrays["y"], selected_oof, fold_mean_oof),
        "selected_vs_historical_prior": paired_patient_test(arrays["patient_id"], arrays["y"], selected_oof, prior_oof),
        "historical_prior_vs_fold_mean": paired_patient_test(arrays["patient_id"], arrays["y"], prior_oof, fold_mean_oof),
    }
    decision = {
        "status": "PASS", "methods_per_inner_screen": 201, "new_methods": 200, "outer_folds": 5,
        "inner_fits": 1005, "outer_selected_fits": 5, "outer_historical_prior_fits": 5,
        "outer_patient_overlap": 0, "inner_outer_test_overlap": 0, "each_segment_predicted_once": True,
        "outer_seed": OUTER_SEED, "selection": selected_rows, "summary": summary, "paired_patient_tests": tests,
        "event_safety": event_safety(arrays["patient_id"], arrays["y"], selected_oof),
        "interpretation_limit": "This estimates the locked 200-method-plus-reference selection procedure under patient-nested resampling. The registry itself was designed after historical interaction with this cohort, so it is not equivalent to a never-seen prospective cohort.",
    }
    np.savez_compressed(OUT / "OOF_PREDICTIONS.npz", y_true=arrays["y"], patient_id=arrays["patient_id"], selected=selected_oof, historical_prior=prior_oof, fold_mean=fold_mean_oof)
    pd.DataFrame([{"model": key, **value} for key, value in summary.items()]).to_csv(OUT / "SUMMARY.csv", index=False)
    (OUT / "DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    journal("nested200_completed", decision=decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
