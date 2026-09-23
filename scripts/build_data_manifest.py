#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collection(path: Path, pattern: str) -> dict:
    files = sorted(path.glob(pattern))
    inventory = hashlib.sha256()
    total = 0
    for item in files:
        size = item.stat().st_size
        total += size
        inventory.update(item.name.encode())
        inventory.update(str(size).encode())
        inventory.update(bytes.fromhex(sha256(item)))
    return {"path": str(path.relative_to(ROOT)), "files": len(files), "bytes": total, "collection_sha256": inventory.hexdigest()}


def npz_summary(path: Path) -> dict:
    values = np.load(path, allow_pickle=True)
    return {
        "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path),
        "arrays": {key: list(values[key].shape) for key in values.files},
        "patients": int(np.unique(values["patient_id"]).size) if "patient_id" in values.files else None,
    }


def main() -> None:
    datasets = {
        "mimic_bp_ppg_raw": collection(ROOT / "data/mimic_bp/raw/ppg", "p*_ppg.npy"),
        "mimic_bp_labels_raw": collection(ROOT / "data/mimic_bp/raw/labels", "p*_labels.npy"),
        "internal_metadata": npz_summary(ROOT / "outputs/neural/cache/metadata.npz"),
        "ppg_bp_219": npz_summary(ROOT / "data/external/prepared/ppg_bp_219.npz"),
        "ambulatory_56": npz_summary(ROOT / "data/external/prepared/ppg_ambulatory_56.npz"),
        "vitaldb_sync128": npz_summary(ROOT / "data/external/vitaldb/prepared/vitaldb_sync128.npz"),
    }
    manifest = {
        "schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exact local snapshot used for the Taqsim manuscript",
        "datasets": datasets,
        "redistribution_warning": "Presence in this local package does not grant public redistribution rights; see DATA_LICENSES.md.",
    }
    destination = ROOT / "data/DATA_MANIFEST.json"
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
