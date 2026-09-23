from __future__ import annotations

import numpy as np
from scipy import signal, stats


EPS = 1e-8


def preprocess_batch(x: np.ndarray, method: str, fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if method == "raw":
        return x
    if method == "robust":
        med = np.median(x, axis=1, keepdims=True)
        q25, q75 = np.quantile(x, [0.25, 0.75], axis=1, keepdims=True)
        return np.clip((x - med) / np.maximum(q75 - q25, 1e-5), -10, 10)
    sos = signal.butter(3, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    source = signal.detrend(x, axis=1, type="linear") if method == "detrend_bandpass" else x
    filtered = signal.sosfiltfilt(sos, source, axis=1)
    if method == "bandpass":
        return filtered
    if method == "detrend_bandpass":
        return filtered
    if method == "bandpass_robust":
        med = np.median(filtered, axis=1, keepdims=True)
        q25, q75 = np.quantile(filtered, [0.25, 0.75], axis=1, keepdims=True)
        return np.clip((filtered - med) / np.maximum(q75 - q25, 1e-5), -10, 10)
    raise ValueError(f"Unknown preprocessing method: {method}")


def _safe_skew(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(stats.skew(x, axis=1, bias=False), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_kurtosis(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(stats.kurtosis(x, axis=1, bias=False), nan=0.0, posinf=0.0, neginf=0.0)


def statistical_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    q = np.quantile(x, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], axis=1).T
    centered = x - x.mean(axis=1, keepdims=True)
    diff = np.diff(x, axis=1)
    t = np.linspace(-1.0, 1.0, x.shape[1])
    slope = (centered @ t) / (np.sum(t * t) + EPS)
    hist_entropy = []
    for row in x:
        hist, _ = np.histogram(row, bins=32, density=False)
        p = hist / max(hist.sum(), 1)
        hist_entropy.append(-np.sum(p[p > 0] * np.log(p[p > 0])))
    values = np.column_stack(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            np.ptp(x, axis=1),
            np.sqrt(np.mean(x * x, axis=1)),
            np.mean(np.abs(centered), axis=1),
            _safe_skew(x),
            _safe_kurtosis(x),
            slope,
            np.mean(np.abs(diff), axis=1),
            np.sqrt(np.mean(diff * diff, axis=1)),
            np.sum(np.abs(diff), axis=1) / x.shape[1],
            np.asarray(hist_entropy),
            q,
        ]
    )
    names = [
        "mean", "std", "min", "max", "range", "rms", "mad_mean", "skew",
        "kurtosis", "linear_slope", "diff_abs_mean", "diff_rms", "line_length",
        "hist_entropy", "q01", "q05", "q10", "q25", "q50", "q75", "q90",
        "q95", "q99",
    ]
    return values.astype(np.float32), names


def spectral_features(x: np.ndarray, fs: int) -> tuple[np.ndarray, list[str]]:
    centered = x - x.mean(axis=1, keepdims=True)
    window = np.hanning(x.shape[1])
    power = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    valid = (freqs >= 0.1) & (freqs <= 15.0)
    pv = power[:, valid]
    fv = freqs[valid]
    total = pv.sum(axis=1) + EPS
    pnorm = pv / total[:, None]
    dominant = fv[np.argmax(pv, axis=1)]
    centroid = np.sum(pnorm * fv[None, :], axis=1)
    bandwidth = np.sqrt(np.sum(pnorm * (fv[None, :] - centroid[:, None]) ** 2, axis=1))
    entropy = -np.sum(pnorm * np.log(pnorm + EPS), axis=1) / np.log(pnorm.shape[1])
    bands = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 15.0)]
    band_values = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_values.append(power[:, mask].sum(axis=1) / total)
    fundamental_power = np.max(pv, axis=1) / total
    values = np.column_stack(
        [dominant, dominant * 60.0, centroid, bandwidth, entropy, fundamental_power, *band_values]
    )
    names = ["dominant_hz", "fft_hr_bpm", "centroid_hz", "bandwidth_hz", "spectral_entropy", "dominant_power_ratio"]
    names += [f"rel_power_{low:g}_{high:g}hz" for low, high in bands]
    return values.astype(np.float32), names


def derivative_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    d1 = np.diff(x, axis=1, prepend=x[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    blocks = []
    names = []
    for tag, z in (("vpg", d1), ("apg", d2)):
        q = np.quantile(z, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=1).T
        blocks.append(
            np.column_stack(
                [z.mean(axis=1), z.std(axis=1), z.min(axis=1), z.max(axis=1),
                 np.sqrt(np.mean(z * z, axis=1)), np.mean(np.abs(z), axis=1),
                 _safe_skew(z), _safe_kurtosis(z), q]
            )
        )
        names += [f"{tag}_{n}" for n in ["mean", "std", "min", "max", "rms", "abs_mean", "skew", "kurtosis", "q01", "q05", "q25", "q50", "q75", "q95", "q99"]]
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def pulse_features(x: np.ndarray, fs: int) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    for row in x:
        scale = max(float(np.std(row)), EPS)
        peaks, props = signal.find_peaks(row, distance=max(1, int(fs * 0.33)), prominence=0.12 * scale)
        troughs, _ = signal.find_peaks(-row, distance=max(1, int(fs * 0.33)), prominence=0.08 * scale)
        intervals = np.diff(peaks) / fs if len(peaks) > 1 else np.array([], dtype=float)
        amplitudes = row[peaks] if len(peaks) else np.array([], dtype=float)
        trough_values = row[troughs] if len(troughs) else np.array([], dtype=float)
        prominences = props.get("prominences", np.array([], dtype=float))
        widths = signal.peak_widths(row, peaks, rel_height=0.5)[0] / fs if len(peaks) else np.array([], dtype=float)
        def sm(a: np.ndarray, fn, default=np.nan) -> float:
            return float(fn(a)) if a.size else float(default)
        hr = 60.0 / sm(intervals, np.mean) if intervals.size and sm(intervals, np.mean) > 0 else np.nan
        rows.append([
            len(peaks), len(troughs), hr,
            sm(intervals, np.mean), sm(intervals, np.std),
            sm(intervals, np.std) / max(sm(intervals, np.mean, 1.0), EPS) if intervals.size else np.nan,
            sm(amplitudes, np.mean), sm(amplitudes, np.std), sm(amplitudes, np.median),
            sm(trough_values, np.mean), sm(trough_values, np.std),
            sm(prominences, np.mean), sm(prominences, np.std),
            sm(widths, np.mean), sm(widths, np.std),
            sm(amplitudes, np.mean) - sm(trough_values, np.mean) if amplitudes.size and trough_values.size else np.nan,
        ])
    names = [
        "n_peaks", "n_troughs", "peak_hr_bpm", "ibi_mean_s", "ibi_std_s", "ibi_cv",
        "peak_mean", "peak_std", "peak_median", "trough_mean", "trough_std",
        "prominence_mean", "prominence_std", "width_mean_s", "width_std_s", "pulse_amplitude",
    ]
    return np.asarray(rows, dtype=np.float32), names


def all_feature_groups(x: np.ndarray, fs: int) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    result: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}
    for family, fn in (
        ("statistics", lambda: statistical_features(x)),
        ("spectral", lambda: spectral_features(x, fs)),
        ("derivative", lambda: derivative_features(x)),
        ("pulse", lambda: pulse_features(x, fs)),
    ):
        result[family], names[family] = fn()
    result["fusion"] = np.concatenate([result[k] for k in ("statistics", "spectral", "derivative", "pulse")], axis=1)
    names["fusion"] = sum((names[k] for k in ("statistics", "spectral", "derivative", "pulse")), [])
    return result, names

