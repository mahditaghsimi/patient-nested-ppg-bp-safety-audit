from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/external"
OUT = DATA_ROOT / "prepared"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_channels(raw: np.ndarray, length: int = 1875) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size < length:
        repeats = int(np.ceil(length / raw.size))
        tiled = np.tile(raw, repeats)[:length]
        # Keep one uninterrupted original record at the exact center so the
        # engine's centered <=2 s crop never crosses a synthetic seam.
        start = (length - raw.size) // 2
        tiled[start : start + raw.size] = raw
        raw = tiled
    elif raw.size > length:
        start = (raw.size - length) // 2
        raw = raw[start : start + length]
    center = np.median(raw)
    scale = max(float(np.quantile(raw, 0.75) - np.quantile(raw, 0.25)), 1e-5)
    ppg = np.clip((raw - center) / scale, -10, 10).astype(np.float32)

    def standardize(values):
        med = np.median(values)
        mad = 1.4826 * np.median(np.abs(values - med))
        if mad < 1e-5:
            mad = max(float(np.std(values)), 1e-5)
        return np.clip((values - med) / mad, -8, 8).astype(np.float32)

    vpg = np.diff(ppg, prepend=ppg[:1])
    apg = np.diff(vpg, prepend=vpg[:1])
    return np.stack([ppg, standardize(vpg), standardize(apg)])


def prepare_ppg_bp() -> dict:
    root = DATA_ROOT / "ppg_bp/raw/Data File"
    table = pd.read_excel(root / "PPG-BP dataset.xlsx", skiprows=1)
    table.columns = [
        "number", "subject_id", "sex", "age", "height", "weight", "sbp", "dbp",
        "heart_rate", "bmi", "hypertension", "diabetes", "cerebral_infarction", "cerebrovascular_disease",
    ]
    table = table[pd.to_numeric(table["subject_id"], errors="coerce").notna()].copy()
    table["subject_id"] = table["subject_id"].astype(int)
    lookup = table.set_index("subject_id")
    rows, signals = [], []
    for path in sorted((root / "0_subject").glob("*.txt"), key=lambda p: tuple(map(int, p.stem.split("_")))):
        subject, record = map(int, path.stem.split("_"))
        if subject not in lookup.index:
            raise KeyError(f"No metadata for PPG-BP subject {subject}")
        raw = np.loadtxt(path, dtype=np.float64)
        resampled = resample_poly(raw, 1, 8)  # 1000 Hz -> 125 Hz
        signals.append(robust_channels(resampled))
        info = lookup.loc[subject]
        rows.append((f"ppgbp_{subject:03d}", record, float(info.sbp), float(info.dbp), float(info.age), str(info.sex)))
    x = np.asarray(signals, dtype=np.float16)
    patient = np.asarray([r[0] for r in rows])
    record = np.asarray([r[1] for r in rows], dtype=np.int16)
    y = np.asarray([[r[2], r[3]] for r in rows], dtype=np.float32)
    age = np.asarray([r[4] for r in rows], dtype=np.float32)
    sex = np.asarray([r[5] for r in rows])
    path = OUT / "ppg_bp_219.npz"
    np.savez_compressed(path, x=x, patient_id=patient, record_id=record, y=y, age=age, sex=sex, fs=125, original_duration_seconds=2.1)
    return {
        "name": "PPG-BP",
        "path": str(path),
        "subjects": int(np.unique(patient).size),
        "records": int(len(patient)),
        "shape": list(x.shape),
        "original_fs": 1000,
        "prepared_fs": 125,
        "original_duration_seconds": 2.1,
        "padding": "periodic context to 15 s storage with one uninterrupted 2.1 s record centered; only <=2 s evaluation is strictly non-padded",
        "label_type": "one cuff SBP/DBP pair shared by three short PPG records",
        "sha256": sha256(path),
    }


def prepare_ppg_56() -> dict:
    root = DATA_ROOT / "ppg_ambulatory_56/raw/PPG_json"
    rows, signals = [], []
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = np.asarray(data["signal"], dtype=np.float64)
        resampled = resample_poly(raw, 5, 4)  # 100 Hz -> 125 Hz
        window = 1875
        for segment, start in enumerate(range(0, len(resampled) - window + 1, window)):
            signals.append(robust_channels(resampled[start : start + window]))
            rows.append((str(data["id"]), segment, float(data["sys_BP"]), float(data["dis_BP"]), float(data["age"]), str(data["gender"])))
    x = np.asarray(signals, dtype=np.float16)
    patient = np.asarray([r[0] for r in rows])
    record = np.asarray([r[1] for r in rows], dtype=np.int16)
    y = np.asarray([[r[2], r[3]] for r in rows], dtype=np.float32)
    age = np.asarray([r[4] for r in rows], dtype=np.float32)
    sex = np.asarray([r[5] for r in rows])
    path = OUT / "ppg_ambulatory_56.npz"
    np.savez_compressed(path, x=x, patient_id=patient, record_id=record, y=y, age=age, sex=sex, fs=125, original_duration_seconds=120.0)
    return {
        "name": "PPG-based-BP-assessment",
        "path": str(path),
        "subjects": int(np.unique(patient).size),
        "records": int(len(patient)),
        "shape": list(x.shape),
        "original_fs": 100,
        "prepared_fs": 125,
        "window_seconds": 15,
        "label_type": "one cuff SBP/DBP pair shared by 7-24 non-overlapping windows per subject (52/56 have eight)",
        "evaluation_unit": "subject; average all window predictions for each subject before computing metrics",
        "sha256": sha256(path),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ppg_bp": prepare_ppg_bp(),
        "ppg_ambulatory_56": prepare_ppg_56(),
        "warning": "These external cuff labels are not simultaneous invasive ABP labels and must be reported as domain-transfer validation, not an interchangeable MIMIC test set.",
    }
    (OUT / "EXTERNAL_DATA_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
