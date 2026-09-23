from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Keep the configuration hash portable across clones. The model resolver maps
# this repository-relative value to the bundled checkpoint even if cwd differs.
PAPAGEI = Path("data/pretraining/papagei_s.pt")


DEFAULT = {
    "view": "derivatives",
    "duration_seconds": 10.0,
    "engineered": "none",
    "head": "direct",
    "loss": "huber",
    "augmentation": "none",
    "balance": "none",
    "dropout": 0.15,
    "head_hidden": 192,
    "epochs": 8,
    "batch_size": 192,
    "learning_rate": 0.001,
    "weight_decay": 0.0002,
}


def with_id(stage: str, number: int, changes: dict) -> dict:
    config = copy.deepcopy(DEFAULT)
    config.update(changes)
    config["stage"] = stage
    stem = f"{stage}_{number:03d}_{config['backbone']}"
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    config["run_id"] = f"{stem}_{digest}"
    return config


def stage_a() -> list[dict]:
    names = ["fcn", "resnet", "tcn", "gated_tcn", "inception", "convnext", "bigru", "patch_transformer", "patch_mixer", "multiscale", "spectral_fusion"]
    rows = [with_id("A", i + 1, {"backbone": name}) for i, name in enumerate(names)]
    rows += [
        with_id("A", 12, {"backbone": "resnet", "engineered": "raw__fusion+scale"}),
        with_id("A", 13, {"backbone": "inception", "engineered": "bandpass_robust__fusion"}),
        with_id("A", 14, {"backbone": "patch_transformer", "engineered": "raw__fusion+scale"}),
        with_id("A", 15, {"backbone": "multiscale", "engineered": "raw__fusion+scale"}),
        with_id("A", 16, {"backbone": "papagei", "view": "ppg_bandpass", "engineered": "none", "foundation_mode": "frozen", "papagei_weights": str(PAPAGEI), "learning_rate": 0.002}),
        with_id("A", 17, {"backbone": "papagei", "view": "ppg_bandpass", "engineered": "raw__fusion+scale", "foundation_mode": "frozen", "papagei_weights": str(PAPAGEI), "learning_rate": 0.002}),
    ]
    return rows


INPUT_VARIANTS = [
    {"duration_seconds": 2.0, "view": "ppg_zscore", "engineered": "none"},
    {"duration_seconds": 5.0, "view": "derivatives", "engineered": "none"},
    {"duration_seconds": 10.0, "view": "ppg_bandpass", "engineered": "none"},
    {"duration_seconds": 10.0, "view": "multiview", "engineered": "none"},
    {"duration_seconds": 15.0, "view": "derivatives", "engineered": "none"},
    {"duration_seconds": 10.0, "view": "derivatives", "engineered": "raw__fusion+scale"},
]


OBJECTIVE_VARIANTS = [
    {"head": "direct", "loss": "l1", "augmentation": "none", "balance": "none"},
    {"head": "direct", "loss": "huber", "augmentation": "jitter_scale", "balance": "none"},
    {"head": "direct", "loss": "tail_huber", "augmentation": "none", "balance": "tail"},
    {"head": "multitask", "loss": "multitask", "augmentation": "jitter_scale", "balance": "none"},
    {"head": "gaussian", "loss": "gaussian_nll", "augmentation": "none", "balance": "none"},
    {"head": "quantile", "loss": "quantile", "augmentation": "mask", "balance": "none"},
]


def write_protocol(path: Path) -> dict:
    protocol = {
        "version": "2.0.0",
        "locked_before_phase2_training": True,
        "selection_unit": "patient",
        "screening_split": "phase1 train versus phase1 validation; prior test excluded from all selection",
        "primary_selection_metric": "mean of SBP/DBP MAE normalized by training-set target SD",
        "tie_breakers": ["mean_mae", "worst_target_mae", "parameter_count"],
        "stage_A": {"configs": stage_a(), "advance_top_unique_backbones": 4},
        "stage_B": {"variants": INPUT_VARIANTS, "advance_top_configs": 4},
        "stage_C": {"variants": OBJECTIVE_VARIANTS, "advance_top_configs": 3},
        "confirmation": "5-fold grouped patient CV on the three finalists; report mean, SD and patient bootstrap CI",
        "external": ["PPG-BP (219 subjects)", "PPG-based-BP-assessment (56 subjects)"],
        "test_policy": "The old internal test was exposed in phase 1 and cannot be called untouched in phase 2.",
        "llm_policy": "No text LLM on raw waveform; test a PPG-native foundation encoder (PaPaGei-S).",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol


if __name__ == "__main__":
    output = ROOT / "phase2/protocol/PROTOCOL_LOCK.json"
    write_protocol(output)
    print(output)
