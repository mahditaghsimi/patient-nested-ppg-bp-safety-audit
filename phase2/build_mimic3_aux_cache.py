from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/mimic3/imports/mimic3_server_bundle_20260820/extracted/phase2/outputs/mimic3_dirty_aware_3gb"
VIEW_CSV = ROOT / "phase2/outputs/mimic3_clean_views/MIMIC3_CLEAN_VIEW_INDEX.csv"
OUT = ROOT / "phase2/outputs/mimic3_aux_cache"
FS = 125
SECONDS = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate(signal: np.ndarray) -> np.ndarray:
    finite = np.isfinite(signal)
    if finite.all():
        return signal
    result = signal.copy()
    index = np.arange(len(signal))
    result[~finite] = np.interp(index[~finite], index[finite], signal[finite])
    return result


def robust_channels(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    center = float(np.median(signal))
    q25, q75 = np.quantile(signal, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-5)
    normalized = np.clip((signal - center) / scale, -10, 10).astype(np.float32)

    def standardize(values: np.ndarray) -> np.ndarray:
        median = float(np.median(values))
        mad = float(1.4826 * np.median(np.abs(values - median)))
        if mad < 1e-5:
            mad = max(float(np.std(values)), 1e-5)
        return np.clip((values - median) / mad, -8, 8).astype(np.float32)

    first = np.diff(normalized, prepend=normalized[:1])
    second = np.diff(first, prepend=first[:1])
    return np.stack([normalized, standardize(first), standardize(second)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    view = pd.read_csv(VIEW_CSV, dtype={"official_subject_id": str, "record": str, "segment": str})
    selected = view[view.view_interpolated].copy().reset_index(drop=True)
    selected["cache_row"] = np.arange(len(selected), dtype=np.int64)
    selected["group_key"] = np.where(
        selected.mapping_status == "official_unique",
        "subject/" + selected.official_subject_id.astype(str).str.zfill(6),
        "record/" + selected.record.astype(str),
    )
    selected.to_csv(OUT / "AUX_METADATA.csv", index=False)
    samples = FS * SECONDS
    destination_path = OUT / "MIMIC3_AUX_15S_3CH_FLOAT16.npy"
    destination = np.lib.format.open_memmap(destination_path, mode="w+", dtype=np.float16, shape=(len(selected), 3, samples))
    by_location = {(row.shard, int(row.shard_row)): int(row.cache_row) for row in selected.itertuples()}
    written = 0
    for shard_path in sorted((DATA / "shards").glob("windows_*.npz")):
        relative = f"shards/{shard_path.name}"
        with np.load(shard_path, allow_pickle=False) as source:
            ppg = source["ppg"].astype(np.float32)
        for shard_row, signal in enumerate(ppg):
            cache_row = by_location.get((relative, shard_row))
            if cache_row is None:
                continue
            signal = interpolate(signal)
            start = (len(signal) - samples) // 2
            destination[cache_row] = robust_channels(signal[start : start + samples]).astype(np.float16)
            written += 1
    destination.flush()
    if written != len(selected) or not np.isfinite(np.asarray(destination)).all():
        raise RuntimeError(f"Cache validation failed: written={written}, selected={len(selected)}")
    manifest = {
        "status": "complete",
        "samples": len(selected),
        "shape": [len(selected), 3, samples],
        "dtype": "float16",
        "source_view": "view_interpolated",
        "central_seconds": SECONDS,
        "normalization": "median/IQR; PPG plus MAD-standardized VPG/APG, matched to target cache",
        "raw_shards_modified": False,
        "view_index_sha256": sha256(VIEW_CSV),
        "cache_sha256": sha256(destination_path),
    }
    (OUT / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
