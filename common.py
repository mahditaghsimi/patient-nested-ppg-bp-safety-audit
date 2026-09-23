from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("data_dir", "labels_dir", "split_path", "forecast_checkpoint"):
        value = Path(cfg[key])
        if not value.is_absolute():
            value = (ROOT / value).resolve()
        cfg[key] = str(value)
    for key in ("cache_path", "schema_path"):
        value = Path(cfg["features"][key])
        if not value.is_absolute():
            value = (ROOT / value).resolve()
        cfg["features"][key] = str(value)
    for key in ("results_path", "registry_path"):
        value = Path(cfg["screening"][key])
        if not value.is_absolute():
            value = (ROOT / value).resolve()
        cfg["screening"][key] = str(value)
    return cfg


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_files(labels_dir: str | Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in Path(labels_dir).rglob("p*_labels.npy"):
        result[path.name.removesuffix("_labels.npy")] = path
    return result


def ppg_files(data_dir: str | Path) -> dict[str, Path]:
    return {
        p.name.removesuffix("_ppg.npy"): p
        for p in sorted(Path(data_dir).glob("p*_ppg.npy"))
    }

