from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2.run_external_validation import infer_checkpoint


DATA = ROOT / "data/external/vitaldb/prepared/vitaldb_sync128.npz"
MANIFEST = ROOT / "data/external/vitaldb/prepared/MANIFEST.json"
MODEL_ROOT = ROOT / "phase2/outputs/external_validation/models"
INTERNAL_META = ROOT / "outputs/neural/cache/metadata.npz"
OUT = ROOT / "phase2/outputs/vitaldb_synchronized_external"


def macro_metrics(patient: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    subjects = np.unique(patient)
    patient_mae = np.asarray([np.abs(pred[patient == subject] - y[patient == subject]).mean(axis=0) for subject in subjects])
    patient_bias = np.asarray([(pred[patient == subject] - y[patient == subject]).mean(axis=0) for subject in subjects])
    result = {
        "patients": int(len(subjects)),
        "windows": int(len(y)),
        "mae_sbp": float(patient_mae[:, 0].mean()),
        "mae_dbp": float(patient_mae[:, 1].mean()),
        "bias_sbp": float(patient_bias[:, 0].mean()),
        "bias_dbp": float(patient_bias[:, 1].mean()),
        "pearson_sbp": float(np.corrcoef(y[:, 0], pred[:, 0])[0, 1]),
        "pearson_dbp": float(np.corrcoef(y[:, 1], pred[:, 1])[0, 1]),
    }
    for target, column in (("sbp", 0), ("dbp", 1)):
        absolute = np.abs(pred[:, column] - y[:, column])
        result[f"rmse_{target}"] = float(np.sqrt(np.mean((pred[:, column] - y[:, column]) ** 2)))
        for limit in (5, 10, 15):
            result[f"pct_{target}_within_{limit}"] = float(100 * np.mean(absolute <= limit))
    return result


def patient_bootstrap(patient: np.ndarray, y: np.ndarray, pred: np.ndarray, iterations: int = 20000) -> dict:
    subjects = np.unique(patient)
    per_patient = np.asarray([np.abs(pred[patient == subject] - y[patient == subject]).mean(axis=0) for subject in subjects])
    rng = np.random.default_rng(20260821)
    means = per_patient[rng.integers(0, len(subjects), size=(iterations, len(subjects)))].mean(axis=1)
    return {
        "mae_sbp_ci_low": float(np.quantile(means[:, 0], 0.025)),
        "mae_sbp_ci_high": float(np.quantile(means[:, 0], 0.975)),
        "mae_dbp_ci_low": float(np.quantile(means[:, 1], 0.025)),
        "mae_dbp_ci_high": float(np.quantile(means[:, 1], 0.975)),
    }


def event_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    event = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    predicted = (pred[:, 0] >= 140) | (pred[:, 1] >= 90)
    score = np.maximum((pred[:, 0] - 140) / 20, (pred[:, 1] - 90) / 10)
    tp = int(np.sum(event & predicted))
    fn = int(np.sum(event & ~predicted))
    tn = int(np.sum(~event & ~predicted))
    fp = int(np.sum(~event & predicted))
    return {
        "high_bp_prevalence": float(event.mean()),
        "direct_sensitivity": float(tp / max(tp + fn, 1)),
        "direct_specificity": float(tn / max(tn + fp, 1)),
        "direct_ppv": float(tp / max(tp + fp, 1)),
        "auprc": float(average_precision_score(event, score)),
        "auroc": float(roc_auc_score(event, score)),
    }


def destination_crossfit(patient: np.ndarray, y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    subjects = np.unique(patient)
    splitter = KFold(5, shuffle=True, random_state=20260821)
    calibrated = np.full_like(pred, np.nan)
    destination_mean = np.full_like(pred, np.nan)
    event_crossfit = np.zeros(len(y), dtype=bool)
    audit = []
    score = np.maximum((pred[:, 0] - 140) / 20, (pred[:, 1] - 90) / 10)
    truth_event = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    for train_subject_i, test_subject_i in splitter.split(subjects):
        train_subjects, test_subjects = subjects[train_subject_i], subjects[test_subject_i]
        overlap = np.intersect1d(train_subjects, test_subjects)
        if overlap.size:
            raise RuntimeError(f"VitalDB destination cross-fit patient overlap: {overlap[:3]}")
        audit.append({"train_subjects": train_subjects.astype(int).tolist(), "test_subjects": test_subjects.astype(int).tolist(), "patient_overlap": 0})
        train = np.isin(patient, train_subjects)
        test = np.isin(patient, test_subjects)
        model = Ridge(alpha=10.0).fit(pred[train], y[train])
        calibrated[test] = model.predict(pred[test])
        destination_mean[test] = y[train].mean(axis=0)
        negative_scores = score[train & ~truth_event]
        threshold = np.quantile(negative_scores, 0.90) if len(negative_scores) else np.inf
        event_crossfit[test] = score[test] >= threshold
    if np.isnan(calibrated).any() or np.isnan(destination_mean).any():
        raise RuntimeError("Incomplete destination cross-fitting")
    return calibrated, destination_mean, event_crossfit, audit


def paired_tests(patient: np.ndarray, y: np.ndarray, candidates: dict[str, np.ndarray], baseline: str) -> list[dict]:
    subjects = np.unique(patient)
    errors = {
        name: np.asarray([np.abs(pred[patient == subject] - y[patient == subject]).mean(axis=0) for subject in subjects])
        for name, pred in candidates.items()
    }
    rows = []
    for name in candidates:
        if name == baseline:
            continue
        for target, column in (("SBP", 0), ("DBP", 1), ("MEAN", None)):
            left = errors[name].mean(axis=1) if column is None else errors[name][:, column]
            right = errors[baseline].mean(axis=1) if column is None else errors[baseline][:, column]
            statistic, p = wilcoxon(left, right)
            rows.append({
                "candidate": name,
                "baseline": baseline,
                "target": target,
                "mean_delta_candidate_minus_baseline": float(np.mean(left - right)),
                "patients_improved_fraction": float(np.mean(left < right)),
                "wilcoxon_statistic": float(statistic),
                "p_unadjusted": float(p),
            })
    order = np.argsort([row["p_unadjusted"] for row in rows])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["p_unadjusted"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running
    return rows


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise RuntimeError("VitalDB preparation gate did not pass")
    with np.load(DATA, allow_pickle=False) as data:
        x = data["x"].copy()
        y = data["y"].copy()
        patient = data["patient_id"].copy()
    checkpoints = sorted(MODEL_ROOT.glob("winner_10s_seed*_fold*/best_model.pt"))
    if len(checkpoints) != 15:
        raise RuntimeError(f"Expected 15 fixed checkpoints, found {len(checkpoints)}")
    OUT.mkdir(parents=True, exist_ok=True)
    member_predictions = np.asarray([infer_checkpoint(path, x) for path in checkpoints])
    zero_shot = member_predictions.mean(axis=0)
    with np.load(INTERNAL_META, allow_pickle=False) as source:
        # Use only the official source-training partition.  The previous
        # implementation averaged train, validation, and test labels while
        # describing the result as a source-training mean.
        train_mask = source["split"].astype(str) == "train"
        source_mean_value = source["y"][train_mask].mean(axis=0)
    source_mean = np.broadcast_to(source_mean_value, y.shape).copy()
    calibrated, destination_mean, crossfit_event_prediction, crossfit_audit = destination_crossfit(patient, y, zero_shot)
    candidates = {
        "mimic_source_mean": source_mean,
        "resnet_zero_shot": zero_shot,
        "destination_mean_crossfit": destination_mean,
        "resnet_affine_crossfit": calibrated,
    }
    summary_rows = []
    for name, prediction in candidates.items():
        row = {"method": name, **macro_metrics(patient, y, prediction), **patient_bootstrap(patient, y, prediction), **event_metrics(y, prediction)}
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    tests = paired_tests(patient, y, candidates, "mimic_source_mean")
    pd.DataFrame(tests).to_csv(OUT / "PAIRED_PATIENT_TESTS.csv", index=False)

    event = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    tn = np.sum(~event & ~crossfit_event_prediction)
    fp = np.sum(~event & crossfit_event_prediction)
    tp = np.sum(event & crossfit_event_prediction)
    fn = np.sum(event & ~crossfit_event_prediction)
    threshold_context = {
        "method": "five-fold patient-cross-fitted threshold on the unchanged zero-shot risk score",
        "target_specificity": 0.90,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }
    decision = {
        "status": "PASS",
        "protocol_gate": manifest,
        "fixed_ensemble_members": len(checkpoints),
        "primary_zero_shot": summary[summary.method == "resnet_zero_shot"].iloc[0].to_dict(),
        "source_mean_baseline": summary[summary.method == "mimic_source_mean"].iloc[0].to_dict(),
        "source_mean_scope": "official_internal_train_partition_only",
        "supervised_context": {
            "destination_mean": summary[summary.method == "destination_mean_crossfit"].iloc[0].to_dict(),
            "affine_calibration": summary[summary.method == "resnet_affine_crossfit"].iloc[0].to_dict(),
            "event_threshold": threshold_context,
        },
        "interpretation": "VitalDB is independent and has synchronized invasive ART labels. Zero-shot is primary; destination calibration is supervised context only.",
    }
    (OUT / "DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        OUT / "PREDICTIONS.npz",
        y_true=y,
        patient_id=patient,
        zero_shot=zero_shot,
        source_mean=source_mean,
        affine_crossfit=calibrated,
        destination_mean=destination_mean,
        member_predictions=member_predictions,
    )
    (OUT / "DESTINATION_CROSSFIT_AUDIT.json").write_text(json.dumps(crossfit_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(threshold_context, indent=2))


if __name__ == "__main__":
    main()
