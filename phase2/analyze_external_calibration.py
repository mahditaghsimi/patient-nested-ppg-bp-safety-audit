from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "phase2/outputs/external_validation"
OUT = EXT / "calibration"


def metrics(y, pred):
    error = pred - y
    return {
        "mae_sbp": float(np.abs(error[:, 0]).mean()), "mae_dbp": float(np.abs(error[:, 1]).mean()),
        "rmse_sbp": float(np.sqrt(np.mean(error[:, 0] ** 2))), "rmse_dbp": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "bias_sbp": float(error[:, 0].mean()), "bias_dbp": float(error[:, 1].mean()),
        "pearson_sbp": float(np.corrcoef(y[:, 0], pred[:, 0])[0, 1]), "pearson_dbp": float(np.corrcoef(y[:, 1], pred[:, 1])[0, 1]),
    }


def crossfit(path: Path, label: str):
    with np.load(path) as data:
        y, pred = data["y_true"], data["y_pred"]
        patient = data["patient_id"].astype(str)
    method_names = ("offset", "affine_ridge", "mean_baseline")
    accumulated = {name: np.zeros_like(pred, dtype=np.float64) for name in method_names}
    counts = np.zeros(len(y), dtype=np.int16)
    # Repeated cross-fitting reduces dependence on one fortunate 5-fold split.
    fold_audit = []
    for repeat_seed in range(10):
        splitter = KFold(5, shuffle=True, random_state=20260820 + repeat_seed)
        for fold, (train, test) in enumerate(splitter.split(y), 1):
            overlap = sorted(set(patient[train]) & set(patient[test]))
            if overlap:
                raise RuntimeError(f"Patient leakage in external adaptation: {overlap[:3]}")
            fold_audit.append({
                "repeat": repeat_seed, "fold": fold,
                "train_patients": patient[train].tolist(), "test_patients": patient[test].tolist(),
                "patient_overlap": 0,
            })
            accumulated["offset"][test] += pred[test] + np.mean(y[train] - pred[train], axis=0)
            accumulated["mean_baseline"][test] += np.broadcast_to(y[train].mean(0), (len(test), 2))
            for target in range(2):
                model = Ridge(alpha=10.0).fit(pred[train], y[train, target])
                accumulated["affine_ridge"][test, target] += model.predict(pred[test])
            counts[test] += 1
    if not np.all(counts == 10):
        raise RuntimeError("Repeated cross-fitting coverage is incomplete")
    methods = {name: (values / counts[:, None]).astype(np.float32) for name, values in accumulated.items()}
    rows = [{"dataset_model": label, "method": "zero_shot", **metrics(y, pred)}]
    rows += [{"dataset_model": label, "method": name, **metrics(y, adjusted)} for name, adjusted in methods.items()]
    np.savez_compressed(OUT / f"crossfit_{label}.npz", patient_id=patient, y_true=y, zero_shot=pred, **methods)
    return rows, fold_audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    audits = {}
    for path, label in [
        (EXT / "predictions_winner_10s_ambulatory56.npz", "winner10s_ambulatory56"),
        (EXT / "predictions_papagei_frozen_10s_ambulatory56.npz", "papagei10s_ambulatory56"),
        (EXT / "predictions_compatible_2s_ambulatory56.npz", "short2s_ambulatory56"),
        (EXT / "predictions_compatible_2s_ppgbp219.npz", "short2s_ppgbp219"),
    ]:
        local_rows, local_audit = crossfit(path, label)
        rows += local_rows
        audits[label] = local_audit
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "crossfit_calibration_metrics.csv", index=False)
    (OUT / "CROSSFIT_FOLD_AUDIT.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
