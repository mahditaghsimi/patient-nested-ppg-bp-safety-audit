from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import label_files, load_config, ppg_files, read_json, write_json
from features.feature_math import all_feature_groups, preprocess_batch


def main() -> None:
    cfg = load_config()
    cache_path = Path(cfg["features"]["cache_path"])
    schema_path = Path(cfg["features"]["schema_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ppg = ppg_files(cfg["data_dir"])
    labels = label_files(cfg["labels_dir"])
    split = read_json(cfg["split_path"])
    ids = sorted(set(ppg) & set(labels))
    if len(ids) != 1524:
        raise RuntimeError(f"Expected 1524 matched patients, found {len(ids)}")
    split_lookup = {pid: name for name in ("train", "validation", "test") for pid in split[name]}
    if set(ids) != set(split_lookup):
        raise RuntimeError("Split patient IDs do not exactly match PPG/label IDs")
    excluded = set(split["excluded_segments"])

    arrays: dict[str, list[np.ndarray]] = {}
    y_parts: list[np.ndarray] = []
    patient_parts: list[np.ndarray] = []
    segment_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    schema: dict[str, dict[str, list[str]]] = {}

    for pid in tqdm(ids, desc="Extracting patient feature banks"):
        raw = np.load(ppg[pid], allow_pickle=False)
        target = np.load(labels[pid], allow_pickle=False)
        if raw.shape != (30, 3750) or target.shape != (30, 2):
            raise ValueError(f"Unexpected shape for {pid}: PPG={raw.shape}, y={target.shape}")
        keep = np.asarray([f"{pid}:{i}" not in excluded for i in range(30)])
        raw = raw[keep]
        target = target[keep]
        seg_ids = np.arange(30, dtype=np.int16)[keep]
        input_samples = 15 * int(cfg["sampling_rate"])
        start = (raw.shape[1] - input_samples) // 2
        raw = raw[:, start : start + input_samples]
        for pre in cfg["features"]["preprocessors"]:
            processed = preprocess_batch(raw, pre, int(cfg["sampling_rate"]))
            groups, names = all_feature_groups(processed, int(cfg["sampling_rate"]))
            schema.setdefault(pre, {})
            for family in cfg["features"]["families"]:
                key = f"{pre}__{family}"
                arrays.setdefault(key, []).append(groups[family])
                schema[pre][family] = names[family]
        y_parts.append(target.astype(np.float32))
        patient_parts.append(np.asarray([pid] * len(target), dtype="U8"))
        segment_parts.append(seg_ids)
        split_parts.append(np.asarray([split_lookup[pid]] * len(target), dtype="U10"))

    payload: dict[str, np.ndarray] = {
        key: np.concatenate(parts, axis=0).astype(np.float32) for key, parts in arrays.items()
    }
    payload["y"] = np.concatenate(y_parts, axis=0)
    payload["patient_id"] = np.concatenate(patient_parts)
    payload["segment_id"] = np.concatenate(segment_parts)
    payload["split"] = np.concatenate(split_parts)
    np.savez_compressed(cache_path, **payload)
    write_json(schema_path, schema)
    write_json(cache_path.parent / "FEATURE_CACHE_MANIFEST.json", {
        "samples": len(payload["y"]),
        "source_window": "central 15 s of each 30-s MIMIC-BP segment (samples 937:2812 at 125 Hz)",
        "feature_scope": "all engineered features are computed from the same central window used by waveform models",
        "learned_scaling": "not applied in cache; fold-specific mean/SD are fitted in phase2.engine.get_engineered on train indices only",
    })
    print(f"Saved {len(payload['y']):,} samples and {len(arrays)} feature matrices to {cache_path}")
    print({name: int(np.sum(payload["split"] == name)) for name in ("train", "validation", "test")})


if __name__ == "__main__":
    main()
