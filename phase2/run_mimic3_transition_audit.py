from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = ROOT / "data/external/mimic3/imports/mimic3_server_bundle_20260820/extracted"
DATA_ROOT = IMPORT_ROOT / "phase2/outputs/mimic3_dirty_aware_3gb"
AUDIT_NPZ = DATA_ROOT / "audit/audit_ppg_abp_001.npz"
ACCEPTED_CSV = DATA_ROOT / "ACCEPTED_WINDOWS.csv"
OUT = ROOT / "phase2/outputs/mimic3_transition_audit_100"
FS = 125.0
SEED = 20260821
TRANSITION_SBP = 10.0
TRANSITION_DBP = 5.0


@dataclass(frozen=True)
class BeatSummary:
    sbp: float
    dbp: float
    map: float
    hr: float
    beats: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_fill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    if not finite.all():
        index = np.arange(x.size)
        x[~finite] = np.interp(index[~finite], index[finite], x[finite])
    return x


def abp_beats(x: np.ndarray, fs: float = FS) -> pd.DataFrame:
    raw = finite_fill(x)
    sos = butter(3, [0.3, 12.0], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, raw)
    prominence = max(5.0, float(np.percentile(filtered, 95) - np.percentile(filtered, 50)) * 0.35)
    peaks, _ = find_peaks(filtered, distance=max(1, int(fs * 60 / 180)), prominence=prominence)
    rows = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        period = (right - left) / fs
        if not 60 / 180 <= period <= 60 / 35:
            continue
        sbp = float(raw[left])
        dbp = float(np.min(raw[left:right]))
        pulse_pressure = sbp - dbp
        if 70 <= sbp <= 220 and 30 <= dbp <= 130 and 15 <= pulse_pressure <= 120:
            rows.append(
                {
                    "peak_sample": int(left),
                    "time_seconds": left / fs,
                    "sbp": sbp,
                    "dbp": dbp,
                    "map": (sbp + 2 * dbp) / 3,
                    "period": period,
                }
            )
    return pd.DataFrame(rows)


def summarize_interval(beats: pd.DataFrame, start: float, stop: float) -> BeatSummary | None:
    part = beats[(beats.time_seconds >= start) & (beats.time_seconds < stop)]
    if len(part) < 4:
        return None
    return BeatSummary(
        sbp=float(part.sbp.median()),
        dbp=float(part.dbp.median()),
        map=float(part["map"].median()),
        hr=float(60 / part.period.median()),
        beats=int(len(part)),
    )


def signal_features(x: np.ndarray, fs: float = FS) -> np.ndarray:
    x = finite_fill(x)
    x = (x - np.median(x)) / max(np.percentile(x, 75) - np.percentile(x, 25), 1e-6)
    dx = np.diff(x, prepend=x[0])
    ddx = np.diff(dx, prepend=dx[0])
    features: list[float] = []
    for values in (x, dx, ddx):
        features.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.percentile(values, 5)),
                float(np.percentile(values, 25)),
                float(np.median(values)),
                float(np.percentile(values, 75)),
                float(np.percentile(values, 95)),
                float(np.mean(np.abs(values))),
                float(np.sqrt(np.mean(values**2))),
            ]
        )
    freqs, power = welch(x, fs=fs, nperseg=min(len(x), int(8 * fs)))
    total = float(power[(freqs >= 0.1) & (freqs <= 20)].sum()) + 1e-12
    for lo, hi in ((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 20.0)):
        features.append(float(power[(freqs >= lo) & (freqs < hi)].sum() / total))
    cardiac = (freqs >= 0.5) & (freqs <= 4.0)
    features.append(float(freqs[cardiac][np.argmax(power[cardiac])]))
    peaks, _ = find_peaks(x, distance=int(fs * 60 / 180), prominence=max(0.1, np.std(x) * 0.25))
    intervals = np.diff(peaks) / fs
    features.extend(
        [
            float(len(peaks)),
            float(np.median(intervals)) if len(intervals) else np.nan,
            float(np.std(intervals)) if len(intervals) else np.nan,
        ]
    )
    return np.asarray(features, dtype=np.float64)


def repeat_last_cycle(history: np.ndarray, future_length: int, fs: float = FS) -> np.ndarray:
    history = finite_fill(history)
    centered = history - np.median(history)
    peaks, _ = find_peaks(centered, distance=int(fs * 60 / 180), prominence=max(5.0, np.std(centered) * 0.25))
    if len(peaks) >= 2:
        period = int(np.clip(np.median(np.diff(peaks[-6:])), fs * 60 / 180, fs * 60 / 35))
    else:
        period = int(fs)
    cycle = history[-period:]
    return np.resize(cycle, future_length)


def mae_pair(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"n": 0, "sbp_mae": np.nan, "dbp_mae": np.nan, "mean_mae": np.nan}
    values = np.abs(y[mask] - prediction[mask]).mean(axis=0)
    return {"n": int(mask.sum()), "sbp_mae": float(values[0]), "dbp_mae": float(values[1]), "mean_mae": float(values.mean())}


def grouped_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str) -> np.ndarray:
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    predictions = np.full_like(y, np.nan, dtype=np.float64)
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        if model_name == "ridge":
            model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0))
        elif model_name == "extra_trees":
            model = make_pipeline(
                SimpleImputer(),
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features=0.75,
                    random_state=SEED + fold,
                    n_jobs=-1,
                ),
            )
        elif model_name == "hist_gradient_boosting":
            model = make_pipeline(
                SimpleImputer(),
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.04,
                        max_leaf_nodes=7,
                        l2_regularization=2.0,
                        random_state=SEED + fold,
                    )
                ),
            )
        else:
            raise ValueError(model_name)
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
    return predictions


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(AUDIT_NPZ, allow_pickle=False) as data:
        ids = data["window_id"].astype(str)
        ppg = data["ppg"].astype(np.float64)
        abp = data["abp"].astype(np.float64)
    accepted = pd.read_csv(ACCEPTED_CSV).set_index("window_id").loc[ids].reset_index()

    rows = []
    feature_rows = []
    waveform_rows = []
    for i, (ppg_window, abp_window) in enumerate(zip(ppg, abp)):
        beats = abp_beats(abp_window)
        first = summarize_interval(beats, 0, 10)
        recent = summarize_interval(beats, 10, 20)
        future = summarize_interval(beats, 20, 30)
        if first is None or recent is None or future is None:
            continue
        delta_sbp = future.sbp - recent.sbp
        delta_dbp = future.dbp - recent.dbp
        transition = abs(delta_sbp) >= TRANSITION_SBP or abs(delta_dbp) >= TRANSITION_DBP
        row = {
            "source_row": i,
            "window_id": ids[i],
            "record": accepted.loc[i, "record"],
            "segment": accepted.loc[i, "segment"],
            "start_sample": int(accepted.loc[i, "start_sample"]),
            "sbp_0_10": first.sbp,
            "dbp_0_10": first.dbp,
            "sbp_10_20": recent.sbp,
            "dbp_10_20": recent.dbp,
            "sbp_20_30": future.sbp,
            "dbp_20_30": future.dbp,
            "delta_sbp": delta_sbp,
            "delta_dbp": delta_dbp,
            "transition": transition,
            "beats_0_10": first.beats,
            "beats_10_20": recent.beats,
            "beats_20_30": future.beats,
        }
        rows.append(row)
        ppg_first = signal_features(ppg_window[: int(10 * FS)])
        ppg_recent = signal_features(ppg_window[int(10 * FS) : int(20 * FS)])
        feature_rows.append(np.concatenate([ppg_first, ppg_recent, ppg_recent - ppg_first]))

        truth_wave = finite_fill(abp_window[int(20 * FS) : int(30 * FS)])
        history = finite_fill(abp_window[: int(20 * FS)])
        wave_predictions = {
            "last_value": np.full_like(truth_wave, history[-1]),
            "repeat_previous_10s": history[-len(truth_wave) :],
            "repeat_last_cycle": repeat_last_cycle(history, len(truth_wave)),
        }
        for name, pred in wave_predictions.items():
            corr = float(np.corrcoef(truth_wave, pred)[0, 1]) if np.std(pred) > 0 else np.nan
            waveform_rows.append(
                {
                    "window_id": ids[i],
                    "record": accepted.loc[i, "record"],
                    "transition": transition,
                    "method": name,
                    "waveform_mae": float(np.mean(np.abs(truth_wave - pred))),
                    "waveform_correlation": corr,
                }
            )

    table = pd.DataFrame(rows)
    features = np.stack(feature_rows)
    y = table[["sbp_20_30", "dbp_20_30"]].to_numpy()
    current = table[["sbp_10_20", "dbp_10_20"]].to_numpy()
    early = table[["sbp_0_10", "dbp_0_10"]].to_numpy()
    groups = table.record.to_numpy()
    transition_mask = table.transition.to_numpy(dtype=bool)
    stable_mask = ~transition_mask

    predictions: dict[str, np.ndarray] = {
        "persistence": current,
        "history_mean": (early + current) / 2,
        "damped_linear_trend": np.clip(current + 0.5 * (current - early), [70, 30], [220, 130]),
        "linear_trend": np.clip(2 * current - early, [70, 30], [220, 130]),
    }
    state_features = np.column_stack([features, early, current, current - early])
    for model_name in ("ridge", "extra_trees", "hist_gradient_boosting"):
        predictions[f"ppg_only_{model_name}"] = grouped_predictions(features, y, groups, model_name)
        predictions[f"ppg_plus_bp_state_{model_name}"] = grouped_predictions(state_features, y, groups, model_name)

    metric_rows = []
    persistence_transition = mae_pair(y, predictions["persistence"], transition_mask)
    for name, pred in predictions.items():
        for subset, mask in (("all", np.ones(len(y), bool)), ("stable", stable_mask), ("transition", transition_mask)):
            values = mae_pair(y, pred, mask)
            reference = persistence_transition if subset == "transition" else mae_pair(y, predictions["persistence"], mask)
            values.update(
                {
                    "method": name,
                    "subset": subset,
                    "skill_vs_persistence": float(1 - values["mean_mae"] / reference["mean_mae"])
                    if reference["mean_mae"] > 0
                    else np.nan,
                }
            )
            if mask.any():
                truth_delta = y[mask] - current[mask]
                pred_delta = pred[mask] - current[mask]
                active = (np.abs(truth_delta[:, 0]) >= TRANSITION_SBP) | (np.abs(truth_delta[:, 1]) >= TRANSITION_DBP)
                values["direction_accuracy"] = (
                    float(np.mean(np.sign(pred_delta[active]) == np.sign(truth_delta[active]))) if active.any() else np.nan
                )
            metric_rows.append(values)

    metrics = pd.DataFrame(metric_rows)
    waveform = pd.DataFrame(waveform_rows)
    table.to_csv(OUT / "transition_windows.csv", index=False)
    metrics.to_csv(OUT / "state_forecast_metrics.csv", index=False)
    waveform.to_csv(OUT / "waveform_baseline_metrics_by_window.csv", index=False)
    waveform_summary = (
        waveform.groupby(["method", "transition"], dropna=False)
        .agg(n=("window_id", "size"), waveform_mae=("waveform_mae", "mean"), waveform_correlation=("waveform_correlation", "mean"))
        .reset_index()
    )
    waveform_summary.to_csv(OUT / "waveform_baseline_summary.csv", index=False)

    valid_records = int(table.record.nunique())
    summary = {
        "status": "exploratory_feasibility_only",
        "source_pairs": int(len(ids)),
        "valid_three_block_windows": int(len(table)),
        "record_groups": valid_records,
        "transition_definition": f"abs(delta_SBP)>={TRANSITION_SBP:g} or abs(delta_DBP)>={TRANSITION_DBP:g} mmHg",
        "transition_windows": int(transition_mask.sum()),
        "transition_fraction": float(transition_mask.mean()),
        "median_abs_delta_sbp": float(np.median(np.abs(table.delta_sbp))),
        "median_abs_delta_dbp": float(np.median(np.abs(table.delta_dbp))),
        "p90_abs_delta_sbp": float(np.percentile(np.abs(table.delta_sbp), 90)),
        "p90_abs_delta_dbp": float(np.percentile(np.abs(table.delta_dbp), 90)),
        "audit_npz_sha256": sha256(AUDIT_NPZ),
        "accepted_csv_sha256": sha256(ACCEPTED_CSV),
        "seed": SEED,
        "warning": "Only 100 audit pairs from 13 record groups; model comparisons are not confirmatory.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = metrics[metrics.subset == "all"].sort_values("mean_mae")
    transitions = metrics[metrics.subset == "transition"].sort_values("mean_mae")
    best_overall = overall.iloc[0]
    best_transition = transitions.iloc[0] if len(transitions) else None
    persistence = overall[overall.method == "persistence"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(table.sbp_10_20, table.sbp_20_30, c=np.where(transition_mask, "tab:red", "tab:blue"), alpha=0.75)
    lo, hi = table[["sbp_10_20", "sbp_20_30"]].to_numpy().min(), table[["sbp_10_20", "sbp_20_30"]].to_numpy().max()
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xlabel="SBP in seconds 10-20", ylabel="SBP in seconds 20-30", title="Future SBP versus persistence")
    show = overall.head(8).sort_values("mean_mae")
    axes[1].barh(show.method, show.mean_mae, color="tab:green")
    axes[1].axvline(persistence.mean_mae, color="black", linestyle="--", label="persistence")
    axes[1].set(xlabel="Mean of SBP/DBP MAE (mmHg)", title="Grouped exploratory forecast")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "transition_feasibility.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    transition_line = (
        f"بهترین روش روی transitionها `{best_transition.method}` با MAE میانگین "
        f"`{best_transition.mean_mae:.2f}` بود."
        if best_transition is not None
        else "transition معتبر کافی مشاهده نشد."
    )
    report = f"""# ممیزی امکان‌پذیری پیش‌بینی ۱۰ ثانیه‌ای فشار خون

وضعیت: **اکتشافی و غیرتأییدی**؛ فقط 100 جفت audit از {valid_records} record در دسترس بود.

## تعریف علی تسک

- ورودی: ثانیه‌های 0 تا 20؛
- مرجع نزدیک: BP در ثانیه‌های 10 تا 20؛
- هدف آینده: BP در ثانیه‌های 20 تا 30؛
- transition: `|ΔSBP| >= {TRANSITION_SBP:g}` یا `|ΔDBP| >= {TRANSITION_DBP:g}` mmHg؛
- split مدل‌های یادگیری‌شونده: GroupKFold بر اساس record، نه window.

## آیا در این نمونه واقعاً تغییر وجود دارد؟

- پنجرهٔ قابل تحلیل: {len(table)} از 100؛
- transition: {int(transition_mask.sum())} ({100 * transition_mask.mean():.1f}%)؛
- میانهٔ |ΔSBP|: {summary['median_abs_delta_sbp']:.2f} mmHg؛
- میانهٔ |ΔDBP|: {summary['median_abs_delta_dbp']:.2f} mmHg؛
- صدک 90 |ΔSBP| / |ΔDBP|: {summary['p90_abs_delta_sbp']:.2f} / {summary['p90_abs_delta_dbp']:.2f} mmHg.

## نتیجهٔ baselineها

- persistence کل: SBP MAE=`{persistence.sbp_mae:.2f}`، DBP MAE=`{persistence.dbp_mae:.2f}`؛
- بهترین نتیجهٔ کلی: `{best_overall.method}` با SBP/DBP MAE=`{best_overall.sbp_mae:.2f}/{best_overall.dbp_mae:.2f}`؛
- {transition_line}

جدول کامل در `state_forecast_metrics.csv` و baselineهای waveform در
`waveform_baseline_summary.csv` ثبت شده‌اند.

## معیار تصمیم

پیش‌بینی ۱۰ ثانیه‌ای فقط وقتی ارزش مقاله‌ای دارد که در دادهٔ بزرگ‌تر و subject-wise:

1. روی transitionها نسبت به persistence skill مثبت و CI بالاتر از صفر داشته باشد؛
2. جهت افزایش/کاهش فشار و event recall را درست گزارش کند؛
3. بهبود فقط ناشی از تکرار سیکل ABP یا phase matching نباشد؛
4. روی cohort و device خارجی نیز ارزیابی شود.

این 100 جفت برای آموزش deep model کافی نیستند؛ خروجی فعلی feasibility gate است، نه ادعای نهایی.
"""
    (OUT / "TRANSITION_AUDIT_FA.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nOVERALL\n", overall.to_string(index=False))
    print("\nTRANSITION\n", transitions.to_string(index=False))


if __name__ == "__main__":
    main()
