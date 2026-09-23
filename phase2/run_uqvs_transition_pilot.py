from __future__ import annotations

import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "phase2" / "external" / "datasets" / "uqvs" / "first10min"
OUT = ROOT / "phase2" / "outputs" / "uqvs_transition_pilot"
FS = 100


def number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_full(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indices = {name: header.index(name) for name in ("RelativeTimeMilliseconds", "NBP (Sys)", "NBP (Dia)", "Pleth")}
        fixed = max(indices.values())
        for row in reader:
            if len(row) <= fixed:
                continue
            rows.append(tuple(number(row[indices[key]]) for key in ("RelativeTimeMilliseconds", "NBP (Sys)", "NBP (Dia)", "Pleth")))
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.isfinite(arr[:, 0])]
    sample = np.rint(arr[:, 0] / 10).astype(int)
    n = int(sample.max()) + 1
    ppg = np.full(n, np.nan, dtype=np.float32)
    sbp = np.full(n, np.nan, dtype=np.float32)
    dbp = np.full(n, np.nan, dtype=np.float32)
    valid = (sample >= 0) & (sample < n)
    ppg[sample[valid]] = arr[valid, 3]
    sbp[sample[valid]] = arr[valid, 1]
    dbp[sample[valid]] = arr[valid, 2]
    return ppg, sbp, dbp


def fill_window(x: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(x)
    if finite.mean() < 0.95 or finite.sum() < 10:
        return None
    index = np.arange(len(x))
    y = np.interp(index, index[finite], x[finite]).astype(np.float64)
    lo, hi = np.percentile(y, [1, 99])
    y = np.clip(y, lo, hi)
    scale = np.median(np.abs(y - np.median(y))) * 1.4826
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    return (y - np.median(y)) / scale


def mean_interval(x: np.ndarray, start_s: int, stop_s: int) -> float:
    values = x[max(0, start_s * FS): min(len(x), (stop_s + 1) * FS)]
    finite = values[np.isfinite(values)]
    if len(finite) < 0.5 * max(1, len(values)):
        return np.nan
    return float(np.mean(finite))


def block_features(x: np.ndarray) -> list[float]:
    d1 = np.diff(x, prepend=x[0])
    d2 = np.diff(d1, prepend=d1[0])
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freq = np.fft.rfftfreq(len(x), 1 / FS)
    pulse_band = (freq >= 0.5) & (freq <= 3.5)
    total_band = (freq >= 0.2) & (freq <= 10)
    peak_freq = float(freq[pulse_band][np.argmax(spectrum[pulse_band])]) if pulse_band.any() else 0.0
    spectral_ratio = float(spectrum[pulse_band].sum() / (spectrum[total_band].sum() + 1e-9))
    peaks, _ = find_peaks(x, distance=int(0.3 * FS), prominence=0.15)
    peak_amp = float(np.median(x[peaks])) if len(peaks) else 0.0
    return [
        float(np.mean(x)), float(np.std(x)), float(np.percentile(x, 75) - np.percentile(x, 25)),
        float(skew(x, bias=False)), float(kurtosis(x, bias=False)),
        float(np.sqrt(np.mean(d1**2))), float(np.sqrt(np.mean(d2**2))),
        float(np.percentile(d1, 95) - np.percentile(d1, 5)), peak_freq, spectral_ratio,
        float(len(peaks)), peak_amp,
    ]


def features(x: np.ndarray) -> np.ndarray:
    blocks = np.array([block_features(block) for block in np.array_split(x, 6)], dtype=np.float64)
    time = np.arange(6, dtype=float)
    slopes = [float(np.polyfit(time, blocks[:, j], 1)[0]) for j in range(blocks.shape[1])]
    vector = np.r_[block_features(x), blocks.ravel(), slopes, blocks[-1] - blocks[0]]
    # Constant or almost-constant subwindows make higher moments undefined.
    # Their absence of shape information is encoded as zero rather than
    # allowing NaNs to leak into estimators that cannot accept them.
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def build_cache() -> tuple[np.ndarray, pd.DataFrame]:
    vectors: list[np.ndarray] = []
    metadata: list[dict] = []
    for path in sorted(DATA.glob("uq_vsd_case*_fulldata_01.csv")):
        case = path.stem.split("case", 1)[1].split("_", 1)[0]
        ppg, sbp, dbp = load_full(path)
        available_seconds = len(ppg) // FS
        for end in range(60, min(280, available_seconds - 320) + 1, 30):
            signal = fill_window(ppg[(end - 60) * FS:end * FS])
            if signal is None:
                continue
            current_sbp = mean_interval(sbp, end - 40, end)
            current_dbp = mean_interval(dbp, end - 40, end)
            future_sbp = mean_interval(sbp, end + 280, end + 320)
            future_dbp = mean_interval(dbp, end + 280, end + 320)
            if not np.all(np.isfinite([current_sbp, current_dbp, future_sbp, future_dbp])):
                continue
            vectors.append(features(signal))
            metadata.append({
                "case": case, "history_end_second": end,
                "current_sbp": current_sbp, "current_dbp": current_dbp,
                "future_sbp": future_sbp, "future_dbp": future_dbp,
                "delta_sbp": future_sbp - current_sbp,
                "delta_dbp": future_dbp - current_dbp,
            })
    if not vectors:
        raise RuntimeError("No valid UQVS PPG forecast samples")
    meta = pd.DataFrame(metadata)
    meta["major_transition"] = (meta.delta_sbp.abs() >= 10) | (meta.delta_dbp.abs() >= 5)
    return np.stack(vectors), meta


def make_model(name: str, seed: int):
    if name.startswith("ridge"):
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name.startswith("histgb"):
        return HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=12, l2_regularization=3.0, random_state=seed
        )
    if name.startswith("extratrees"):
        return ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.7,
            n_jobs=-1, random_state=seed,
        )
    raise KeyError(name)


def evaluate_protocol(x: np.ndarray, meta: pd.DataFrame, splits, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    prediction_rows: list[dict] = []
    y = meta[["delta_sbp", "delta_dbp"]].to_numpy(float)
    current = meta[["current_sbp", "current_dbp"]].to_numpy(float)
    future = meta[["future_sbp", "future_dbp"]].to_numpy(float)
    event = meta.major_transition.to_numpy(bool)
    for fold, (train, test) in enumerate(splits):
        augmented_train = np.c_[x[train], current[train]]
        augmented_test = np.c_[x[test], current[test]]
        candidates = [
            ("persistence", "none"),
            ("ridge_delta", "delta"),
            ("histgb_delta", "delta"),
            ("extratrees_delta", "delta"),
            ("extratrees_eventweighted_delta", "delta_weighted"),
            ("ridge_ppg_absolute", "absolute"),
            ("extratrees_ppg_absolute", "absolute"),
        ]
        for name, mode in candidates:
            pred_delta = np.zeros((len(test), 2), dtype=float)
            pred = current[test].copy()
            if mode in {"delta", "delta_weighted"}:
                for target in range(2):
                    local = make_model(name, 1000 * target + 100 + fold)
                    fit_kwargs = {}
                    if mode == "delta_weighted":
                        fit_kwargs["sample_weight"] = 1.0 + 3.0 * event[train].astype(float)
                    local.fit(augmented_train, y[train, target], **fit_kwargs)
                    pred_delta[:, target] = local.predict(augmented_test)
                pred = current[test] + pred_delta
            elif mode == "absolute":
                for target in range(2):
                    local = make_model(name, 1000 * target + 100 + fold)
                    local.fit(x[train], future[train, target])
                    pred[:, target] = local.predict(x[test])
                pred_delta = pred - current[test]
            if protocol == "patient_group_5fold":
                for local_index, source_index in enumerate(test):
                    prediction_rows.append({
                        "sample_index": int(source_index), "case": meta.iloc[source_index].case,
                        "fold": fold, "model": name,
                        "major_transition": bool(event[source_index]),
                        "current_sbp": float(current[source_index, 0]),
                        "current_dbp": float(current[source_index, 1]),
                        "future_sbp": float(future[source_index, 0]),
                        "future_dbp": float(future[source_index, 1]),
                        "pred_sbp": float(pred[local_index, 0]),
                        "pred_dbp": float(pred[local_index, 1]),
                    })
            for subset, mask in {"all": np.ones(len(test), bool), "major_transition": event[test]}.items():
                if not mask.any():
                    continue
                mae = np.mean(np.abs(pred[mask] - future[test][mask]), axis=0)
                base = np.mean(np.abs(current[test][mask] - future[test][mask]), axis=0)
                true_direction = np.where(y[test, 0] <= -10, -1, np.where(y[test, 0] >= 10, 1, 0))
                pred_direction = np.where(pred_delta[:, 0] <= -10, -1, np.where(pred_delta[:, 0] >= 10, 1, 0))
                rows.append({
                    "protocol": protocol, "fold": fold, "model": name, "subset": subset,
                    "n": int(mask.sum()), "mae_sbp": float(mae[0]), "mae_dbp": float(mae[1]),
                    "skill_sbp_vs_persistence": float(1 - mae[0] / (base[0] + 1e-9)),
                    "skill_dbp_vs_persistence": float(1 - mae[1] / (base[1] + 1e-9)),
                    "sbp_direction_balanced_accuracy": float(balanced_accuracy_score(true_direction[mask], pred_direction[mask])),
                })
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, meta = build_cache()
    np.savez_compressed(OUT / "PPG_FEATURE_CACHE.npz", x=x)
    meta.to_csv(OUT / "PPG_FORECAST_METADATA.csv", index=False)
    random = ShuffleSplit(n_splits=20, test_size=0.2, random_state=42).split(x)
    grouped = GroupKFold(n_splits=5).split(x, groups=meta.case)
    random_results, _ = evaluate_protocol(x, meta, random, "sample_random_80_20")
    group_results, group_predictions = evaluate_protocol(x, meta, grouped, "patient_group_5fold")
    results = pd.concat([random_results, group_results], ignore_index=True)
    results.to_csv(OUT / "FOLD_RESULTS.csv", index=False)
    group_predictions.to_csv(OUT / "GROUP_OOF_PREDICTIONS.csv", index=False)
    summary = results.groupby(["protocol", "model", "subset"]).agg(
        folds=("fold", "nunique"), n_mean=("n", "mean"),
        mae_sbp_mean=("mae_sbp", "mean"), mae_sbp_sd=("mae_sbp", "std"),
        mae_dbp_mean=("mae_dbp", "mean"), mae_dbp_sd=("mae_dbp", "std"),
        skill_sbp_mean=("skill_sbp_vs_persistence", "mean"),
        skill_dbp_mean=("skill_dbp_vs_persistence", "mean"),
        direction_bacc_mean=("sbp_direction_balanced_accuracy", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    cost_rows = []
    y_delta = meta[["delta_sbp", "delta_dbp"]].to_numpy(float)
    y_absolute = meta[["future_sbp", "future_dbp"]].to_numpy(float)
    current = meta[["current_sbp", "current_dbp"]].to_numpy(float)
    for name, mode in [
        ("ridge_delta", "delta"), ("histgb_delta", "delta"),
        ("extratrees_delta", "delta"),
        ("extratrees_eventweighted_delta", "delta_weighted"),
        ("ridge_ppg_absolute", "absolute"),
        ("extratrees_ppg_absolute", "absolute"),
    ]:
        input_x = np.c_[x, current] if mode.startswith("delta") else x
        target_y = y_delta if mode.startswith("delta") else y_absolute
        fitted = []
        for target in range(2):
            model = make_model(name, 9000 + target)
            kwargs = {}
            if mode == "delta_weighted":
                kwargs["sample_weight"] = 1.0 + 3.0 * meta.major_transition.to_numpy(float)
            model.fit(input_x, target_y[:, target], **kwargs)
            fitted.append(model)
        payload_bytes = sum(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) for model in fitted)
        probe = np.tile(input_x[: min(64, len(input_x))], (16, 1))
        start = time.perf_counter()
        for model in fitted:
            model.predict(probe)
        elapsed = time.perf_counter() - start
        node_count = 0
        for model in fitted:
            if hasattr(model, "estimators_"):
                node_count += sum(tree.tree_.node_count for tree in model.estimators_)
        cost_rows.append({
            "model": name, "serialized_bytes_two_targets": payload_bytes,
            "tree_nodes_two_targets": node_count,
            "inference_microseconds_per_sample_two_targets": elapsed * 1e6 / len(probe),
            "input_features": int(input_x.shape[1]),
        })
    pd.DataFrame(cost_rows).to_csv(OUT / "MODEL_COST.csv", index=False)
    decision = {
        "cases": int(meta.case.nunique()), "samples": int(len(meta)), "features": int(x.shape[1]),
        "major_transition_count": int(meta.major_transition.sum()),
        "major_transition_prevalence": float(meta.major_transition.mean()),
        "interpretation_gate": "A candidate is promising only if patient-group transition skill is positive for both SBP and DBP.",
    }
    (OUT / "DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
