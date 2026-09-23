from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy"
META = ROOT / "outputs/neural/cache/metadata.npz"
OUT = ROOT / "phase2/outputs/clinical_sqi"


def percentile_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(ordered, values, side="right") / max(len(ordered), 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(CACHE, mmap_mode="r", allow_pickle=False)
    with np.load(META, allow_pickle=False) as meta:
        split = meta["split"].astype(str)
        patient = meta["patient_id"].astype(str)
    n, _, length = x.shape
    sampling_rate = 125
    periodicity = np.empty(n, np.float32)
    spectral_concentration = np.empty(n, np.float32)
    flat_fraction = np.empty(n, np.float32)
    spike_ratio = np.empty(n, np.float32)
    finite_fraction = np.empty(n, np.float32)
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        signal = np.asarray(x[start:stop, 0], dtype=np.float32)
        finite_fraction[start:stop] = np.isfinite(signal).mean(1)
        signal = np.nan_to_num(signal)
        signal -= signal.mean(1, keepdims=True)
        diff = np.abs(np.diff(signal, axis=1))
        flat_fraction[start:stop] = (diff < 1e-4).mean(1)
        spike_ratio[start:stop] = np.quantile(diff, 0.995, axis=1) / (np.median(diff, axis=1) + 1e-4)
        spectrum = np.abs(np.fft.rfft(signal, axis=1)) ** 2
        frequency = np.fft.rfftfreq(length, 1 / sampling_rate)
        physiological = (frequency >= 0.5) & (frequency <= 5.0)
        relevant = (frequency >= 0.1) & (frequency <= 20.0)
        spectral_concentration[start:stop] = spectrum[:, physiological].sum(1) / (spectrum[:, relevant].sum(1) + 1e-8)
        # FFT autocorrelation gives a fast periodicity proxy over plausible HR
        # lags (roughly 40--180 bpm).
        autocorrelation = np.fft.irfft(spectrum, n=length, axis=1)
        autocorrelation /= np.maximum(autocorrelation[:, :1], 1e-8)
        periodicity[start:stop] = autocorrelation[:, 42:188].max(1)

    train = split == "train"
    periodicity_rank = percentile_rank(periodicity[train], periodicity)
    spectral_rank = percentile_rank(spectral_concentration[train], spectral_concentration)
    flat_bad_rank = percentile_rank(flat_fraction[train], flat_fraction)
    spike_bad_rank = percentile_rank(spike_ratio[train], spike_ratio)
    quality = (
        0.45 * periodicity_rank
        + 0.35 * spectral_rank
        + 0.10 * (1 - flat_bad_rank)
        + 0.10 * (1 - spike_bad_rank)
    )
    quality *= finite_fraction
    quality = np.clip(quality, 0, 1).astype(np.float32)
    np.savez_compressed(
        OUT / "clinical_sqi.npz",
        quality_score=quality,
        periodicity=periodicity,
        spectral_concentration=spectral_concentration,
        flat_fraction=flat_fraction,
        spike_ratio=spike_ratio,
        finite_fraction=finite_fraction,
        patient_id=patient,
        split=split,
    )
    frame = pd.DataFrame({
        "split": split,
        "quality": quality,
        "periodicity": periodicity,
        "spectral_concentration": spectral_concentration,
        "flat_fraction": flat_fraction,
        "spike_ratio": spike_ratio,
    })
    summary = frame.groupby("split").agg(["count", "mean", "median", "std"]).round(6)
    summary.to_csv(OUT / "SQI_SUMMARY.csv")
    decision = {
        "samples": int(n),
        "definition": "label-free composite rank: periodicity 45%, physiological spectral concentration 35%, non-flatness 10%, non-spikiness 10%",
        "train_quality_quantiles": np.quantile(quality[train], [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]).tolist(),
        "leakage_policy": "quality uses waveform only; percentile reference fitted on train split",
    }
    (OUT / "SQI_MANIFEST.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
