from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, ppg_files, write_json


def robust_channels(raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Reproduce the three forecasting-pretext input channels exactly."""
    raw64 = np.asarray(raw, dtype=np.float64)
    center = float(np.median(raw64))
    q25, q75 = np.quantile(raw64, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-5)
    x = np.clip((raw64 - center) / scale, -10.0, 10.0).astype(np.float32)

    def robust_standardize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        median = float(np.median(values))
        mad = float(1.4826 * np.median(np.abs(values - median)))
        if mad < 1e-5:
            mad = max(float(values.std()), 1e-5)
        return np.clip((values - median) / mad, -8.0, 8.0).astype(np.float32)

    dx = np.diff(x, prepend=x[:1])
    ddx = np.diff(dx, prepend=dx[:1])
    return np.stack([x, robust_standardize(dx), robust_standardize(ddx)]), center, scale


def main() -> None:
    cfg = load_config()
    output_dir = Path("outputs/neural/cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    x_path = output_dir / "ppg15s_3ch_float16.npy"
    metadata_path = output_dir / "metadata.npz"
    with np.load(cfg["features"]["cache_path"], allow_pickle=False) as data:
        patient = data["patient_id"].copy()
        segment = data["segment_id"].copy()
        y = data["y"].copy()
        split = data["split"].copy()
    n = len(patient)
    input_samples = 15 * int(cfg["sampling_rate"])
    destination = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float16, shape=(n, 3, input_samples)
    )
    centers = np.empty(n, dtype=np.float32)
    scales = np.empty(n, dtype=np.float32)
    files = ppg_files(cfg["data_dir"])
    current_pid = None
    current_array = None
    for index in tqdm(range(n), desc="Building neural input cache"):
        pid = str(patient[index])
        if pid != current_pid:
            current_pid = pid
            current_array = np.load(files[pid], mmap_mode="r", allow_pickle=False)
        segment_samples = int(current_array.shape[1])
        start = (segment_samples - input_samples) // 2
        raw = np.asarray(current_array[int(segment[index]), start : start + input_samples], dtype=np.float32)
        channels, center, scale = robust_channels(raw)
        destination[index] = channels.astype(np.float16)
        centers[index], scales[index] = center, scale
    destination.flush()
    np.savez_compressed(
        metadata_path,
        patient_id=patient,
        segment_id=segment,
        split=split,
        y=y,
        raw_center=centers,
        raw_iqr=scales,
    )
    write_json(
        output_dir / "manifest.json",
        {
            "samples": n,
            "shape": [n, 3, input_samples],
            "dtype": "float16",
            "source_window": "central 15 s of each 30-s MIMIC-BP segment (samples 937:2812 at 125 Hz)",
            "normalization": "per-window median/IQR; PPG, VPG, APG channels; no cohort-fitted waveform transform",
            "test_used_for_training_or_selection": False,
        },
    )
    print(f"Saved {x_path} and {metadata_path}")


if __name__ == "__main__":
    main()
