from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2.engine import load_arrays, metrics, run_experiment
from phase2.run_confirmation_cv import subject_strata


OUT = ROOT / "phase2/outputs/reproducibility_confirmation_20260821"
LOCK = ROOT / "phase2/protocol/REPRODUCIBILITY_CONFIRMATION_LOCK_20260821.json"


def patient_errors(patient: np.ndarray, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.abs(pred[patient == subject] - y[patient == subject]).mean(axis=0)
        for subject in np.unique(patient)
    ])


def paired_bootstrap(patient_error_delta: np.ndarray, iterations: int = 20000) -> dict:
    rng = np.random.default_rng(314159)
    n = len(patient_error_delta)
    sampled = patient_error_delta[rng.integers(0, n, size=(iterations, n))].mean(axis=1)
    return {
        "mean_delta": float(patient_error_delta.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def main() -> None:
    protocol = json.loads(LOCK.read_text(encoding="utf-8"))
    if not protocol.get("locked_before_training"):
        raise RuntimeError("The reproducibility protocol was not locked before training")

    arrays = load_arrays()
    y = arrays["y"].astype(np.float32)
    patient = arrays["patient_id"].astype(str)
    subjects, strata = subject_strata(patient, y)
    seed = int(protocol["split"]["seed"])
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    oof_model = np.full_like(y, np.nan)
    oof_mean = np.full_like(y, np.nan)
    seen = np.zeros(len(y), dtype=np.int8)

    for fold, (train_subject_i, test_subject_i) in enumerate(splitter.split(subjects, strata), 1):
        train_subjects = subjects[train_subject_i]
        test_subjects = subjects[test_subject_i]
        if np.intersect1d(train_subjects, test_subjects).size:
            raise RuntimeError(f"Patient leakage detected in fold {fold}")
        train_idx = np.flatnonzero(np.isin(patient, train_subjects))
        test_idx = np.flatnonzero(np.isin(patient, test_subjects))
        if np.any(seen[test_idx]):
            raise RuntimeError(f"Repeated evaluation samples in fold {fold}")

        config = dict(protocol["fixed_model"])
        config.update({
            "stage": "REPRO_CONFIRM",
            "run_id": f"REPRO_seed{seed}_fold{fold}",
            "fixed_epochs": True,
        })
        run_dir = OUT / "runs" / config["run_id"]
        result = run_experiment(config, arrays, train_idx, test_idx, run_dir, seed=seed + fold, save_model=True)
        with np.load(run_dir / "best_predictions.npz", allow_pickle=False) as prediction_file:
            if not np.array_equal(prediction_file["indices"], test_idx):
                raise RuntimeError(f"Prediction-index mismatch in fold {fold}")
            oof_model[test_idx] = prediction_file["y_pred"]
        oof_mean[test_idx] = y[train_idx].mean(axis=0)
        seen[test_idx] = 1
        result.update({
            "fold": fold,
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "patient_overlap": 0,
            "checkpoint_bytes": (run_dir / "best_model.pt").stat().st_size,
        })
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / "fold_results.csv", index=False)

    if not np.all(seen == 1) or np.isnan(oof_model).any() or np.isnan(oof_mean).any():
        raise RuntimeError("OOF coverage is incomplete")

    global_std = y.std(axis=0)
    model_metrics = metrics(y, oof_model, global_std)
    mean_metrics = metrics(y, oof_mean, global_std)
    model_patient = patient_errors(patient, y, oof_model)
    mean_patient = patient_errors(patient, y, oof_mean)
    tests = {}
    for name, column in (("SBP", 0), ("DBP", 1), ("MEAN", None)):
        left = model_patient.mean(axis=1) if column is None else model_patient[:, column]
        right = mean_patient.mean(axis=1) if column is None else mean_patient[:, column]
        delta = left - right
        statistic, p_value = wilcoxon(left, right)
        tests[name] = {
            **paired_bootstrap(delta),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p": float(p_value),
            "patients_improved_fraction": float(np.mean(delta < 0)),
        }

    prior = {"mae_sbp": 13.139089202880859, "mae_dbp": 8.343780326843262}
    success = bool(
        model_metrics["mae_sbp"] < mean_metrics["mae_sbp"]
        and model_metrics["mae_dbp"] < mean_metrics["mae_dbp"]
        and len(rows) == 5
        and all(row["patient_overlap"] == 0 for row in rows)
    )
    decision = {
        "status": "PASS" if success else "FAIL",
        "purpose": "independent reproducibility check; not primary model selection",
        "samples": int(len(y)),
        "subjects": int(len(subjects)),
        "folds": 5,
        "model_oof": model_metrics,
        "population_mean_oof": mean_metrics,
        "paired_patient_tests_model_minus_mean": tests,
        "prior_same_family_15_fit_mean": prior,
        "absolute_difference_from_prior": {
            "mae_sbp": float(abs(model_metrics["mae_sbp"] - prior["mae_sbp"])),
            "mae_dbp": float(abs(model_metrics["mae_dbp"] - prior["mae_dbp"])),
        },
        "checkpoints_saved": 5,
        "claim_limit": "This confirms executable patient-separated training, not clinical validation or device interchangeability."
    }
    np.savez_compressed(OUT / "oof_predictions.npz", y_true=y, patient_id=patient, model=oof_model, population_mean=oof_mean)
    (OUT / "DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
