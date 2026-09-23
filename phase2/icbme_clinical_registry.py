from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT = {
    "backbone": "resnet",
    "view": "multiview",
    "duration_seconds": 10.0,
    "engineered": "none",
    "head": "direct",
    "loss": "huber",
    "augmentation": "none",
    "balance": "none",
    "dropout": 0.15,
    "head_hidden": 192,
    "epochs": 5,
    "patience": 2,
    "batch_size": 192,
    "learning_rate": 1e-3,
    "weight_decay": 2e-4,
}


def make(stage: str, number: int, label: str, changes: dict) -> dict:
    config = copy.deepcopy(DEFAULT)
    config.update(changes)
    config["stage"] = stage
    config["clinical_method"] = label
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    config["run_id"] = f"{stage}_{number:03d}_{label}_{digest}"
    return config


BASES = {
    "resnet_multiview10": {"backbone": "resnet", "view": "multiview", "duration_seconds": 10.0},
    "resnet_derivative5": {"backbone": "resnet", "view": "derivatives", "duration_seconds": 5.0},
}


OBJECTIVES = [
    ("huber", {"head": "direct", "loss": "huber"}),
    ("l1", {"head": "direct", "loss": "l1"}),
    ("logcosh", {"head": "direct", "loss": "logcosh"}),
    ("ccc05", {"head": "direct", "loss": "huber_ccc", "ccc_weight": 0.05}),
    ("ccc10", {"head": "direct", "loss": "huber_ccc", "ccc_weight": 0.10}),
    ("ccc20", {"head": "direct", "loss": "huber_ccc", "ccc_weight": 0.20}),
    ("asym25", {"head": "direct", "loss": "asymmetric_high", "underestimate_penalty": 0.25}),
    ("asym50", {"head": "direct", "loss": "asymmetric_high", "underestimate_penalty": 0.50}),
    ("asym75", {"head": "direct", "loss": "asymmetric_high", "underestimate_penalty": 0.75}),
    ("asym100", {"head": "direct", "loss": "asymmetric_high", "underestimate_penalty": 1.00}),
    ("focal25", {"head": "direct", "loss": "focal_huber", "focal_weight": 0.25}),
    ("focal50", {"head": "direct", "loss": "focal_huber", "focal_weight": 0.50}),
    ("physio05", {"head": "direct", "loss": "physiology_huber", "physiology_weight": 0.05}),
    ("physio15", {"head": "direct", "loss": "physiology_huber", "physiology_weight": 0.15}),
    ("clinical10", {"head": "clinical", "loss": "huber", "clinical_head_weight": 0.10}),
    ("clinical25", {"head": "clinical", "loss": "huber", "clinical_head_weight": 0.25}),
    ("ordinal10", {"head": "ordinal", "loss": "huber", "ordinal_head_weight": 0.10}),
    ("ordinal25", {"head": "ordinal", "loss": "huber", "ordinal_head_weight": 0.25}),
    ("physio_multitask", {"head": "multitask", "loss": "multitask"}),
    ("physio_multitask_aug", {"head": "multitask", "loss": "multitask", "augmentation": "jitter_scale"}),
]


def objective_configs() -> list[dict]:
    configs, number = [], 1
    for base_name, base in BASES.items():
        for objective_name, objective in OBJECTIVES:
            configs.append(make("MEDOBJ", number, f"{base_name}__{objective_name}", {**base, **objective}))
            number += 1
    return configs


def robustness_configs() -> list[dict]:
    augmentations = ["none", "jitter_scale", "mask", "baseline_wander", "motion_spike", "time_shift", "channel_dropout", "compound_clinical"]
    configs, number = [], 1
    for base_name, base in BASES.items():
        for augmentation in augmentations:
            configs.append(make("ROBUST", number, f"{base_name}__aug_{augmentation}", {
                **base, "head": "multitask", "loss": "multitask", "augmentation": augmentation,
            }))
            number += 1
    return configs


def balance_configs() -> list[dict]:
    variants = [
        ("none", {}),
        ("tail15", {"balance": "tail", "tail_sampling_weight": 1.5}),
        ("tail20", {"balance": "tail", "tail_sampling_weight": 2.0}),
        ("tail30", {"balance": "tail", "tail_sampling_weight": 3.0}),
        ("sbp_quantile", {"balance": "sbp_quantile"}),
        ("clinical_stage", {"balance": "clinical_stage"}),
        ("quality", {"balance": "quality"}),
        ("quality_tail", {"balance": "quality_tail", "tail_sampling_weight": 2.0}),
    ]
    configs, number = [], 1
    for base_name, base in BASES.items():
        for variant_name, variant in variants:
            configs.append(make("BALANCE", number, f"{base_name}__{variant_name}", {
                **base, "head": "multitask", "loss": "multitask", "augmentation": "jitter_scale", **variant,
            }))
            number += 1
    return configs


def representation_configs() -> list[dict]:
    variants = [
        ("resnet_ppg2", {"backbone": "resnet", "view": "ppg_zscore", "duration_seconds": 2.0}),
        ("resnet_deriv5", {"backbone": "resnet", "view": "derivatives", "duration_seconds": 5.0}),
        ("resnet_deriv10", {"backbone": "resnet", "view": "derivatives", "duration_seconds": 10.0}),
        ("resnet_multi10", {"backbone": "resnet", "view": "multiview", "duration_seconds": 10.0}),
        ("resnet_deriv15", {"backbone": "resnet", "view": "derivatives", "duration_seconds": 15.0}),
        ("gated_multi10", {"backbone": "gated_tcn", "view": "multiview", "duration_seconds": 10.0}),
        ("gated_deriv15", {"backbone": "gated_tcn", "view": "derivatives", "duration_seconds": 15.0}),
        ("spectral_deriv10", {"backbone": "spectral_fusion", "view": "derivatives", "duration_seconds": 10.0}),
        ("spectral_multi10", {"backbone": "spectral_fusion", "view": "multiview", "duration_seconds": 10.0}),
        ("multiscale_deriv10", {"backbone": "multiscale", "view": "derivatives", "duration_seconds": 10.0}),
        ("inception_deriv10", {"backbone": "inception", "view": "derivatives", "duration_seconds": 10.0}),
        ("patch_deriv10", {"backbone": "patch_transformer", "view": "derivatives", "duration_seconds": 10.0}),
    ]
    return [make("REP", i, label, {
        **variant, "head": "multitask", "loss": "multitask", "augmentation": "jitter_scale",
    }) for i, (label, variant) in enumerate(variants, 1)]


def optimization_configs() -> list[dict]:
    variants = [
        ("drop05", {"dropout": 0.05}), ("drop10", {"dropout": 0.10}),
        ("drop25", {"dropout": 0.25}), ("drop35", {"dropout": 0.35}),
        ("hidden96", {"head_hidden": 96}), ("hidden320", {"head_hidden": 320}),
        ("lr3e4", {"learning_rate": 3e-4}), ("lr6e4", {"learning_rate": 6e-4}),
        ("wd1e5", {"weight_decay": 1e-5}), ("wd1e3", {"weight_decay": 1e-3}),
        ("batch128", {"batch_size": 128}), ("batch256", {"batch_size": 256}),
    ]
    base = {**BASES["resnet_multiview10"], "head": "multitask", "loss": "multitask", "augmentation": "jitter_scale"}
    return [make("OPT", i, label, {**base, **variant}) for i, (label, variant) in enumerate(variants, 1)]


def all_configs() -> list[dict]:
    return objective_configs() + robustness_configs() + balance_configs() + representation_configs() + optimization_configs()


def protocol() -> dict:
    configs = all_configs()
    return {
        "version": "1.0.0",
        "locked_before_training": True,
        "purpose": "ICBME medical-method sprint: clinically stratified BP estimation, robustness and safe deployment",
        "selection_data": "MIMIC-BP train versus validation; old internal test excluded",
        "selection_unit_policy": "predefined subject-disjoint split",
        "primary_metric": "clinical_selection_score = 0.75 normalized overall MAE + 0.25 normalized tail MAE",
        "safety_gate": "candidate overall mean MAE may not exceed fixed baseline by >0.10 mmHg",
        "confirmation": "fixed baseline plus two best novel candidates; 5 patient-wise folds x 3 seeds; fixed epoch budget",
        "medical_metrics": ["SBP/DBP MAE", "tail MAE", "hypo/hypertension MAE", "high-BP sensitivity/specificity", "BHS <=5/10/15", "bias", "correlation"],
        "external_policy": "only confirmed winner; zero-shot separated from cross-fitted local calibration",
        "config_count": len(configs),
        "configs": configs,
    }


def main() -> None:
    path = ROOT / "phase2/protocol/ICBME_CLINICAL_SEARCH_LOCK.json"
    path.write_text(json.dumps(protocol(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(f"configs={len(all_configs())}")


if __name__ == "__main__":
    main()
