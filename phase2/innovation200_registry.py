from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "phase2/protocol/INNOVATION200_SEARCH_LOCK.json"


DEFAULT = {
    "backbone": "adaptive_resnet",
    "view": "multiview_zscore",
    "duration_seconds": 10.0,
    "engineered": "none",
    "head": "multitask",
    "loss": "multitask",
    "augmentation": "jitter_scale",
    "balance": "none",
    "model_width": 96,
    "model_depth": 6,
    "model_kernel": 5,
    "stem_kernel": 11,
    "stem_stride": 4,
    "dilation_mode": "cycle",
    "channel_attention": "eca",
    "pooling": "stats",
    "block_dropout": 0.0,
    "dropout": 0.15,
    "head_hidden": 192,
    "huber_beta": 0.5,
    "aux_weight": 0.35,
    "consistency_weight": 0.10,
    "target_weights": [1.0, 1.0],
    "optimizer": "adamw",
    "scheduler": "none",
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
    config["innovation_label"] = label
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    config["run_id"] = f"I200_{stage}_{number:03d}_{digest}"
    return config


def architecture_configs() -> list[dict]:
    rows: list[dict] = []
    number = 1
    for width in (64, 96, 128, 160):
        for depth in (4, 6, 8):
            for attention in ("none", "eca", "se"):
                rows.append(make("ARCH", number, f"adaptive_w{width}_d{depth}_{attention}_stats", {
                    "model_width": width, "model_depth": depth,
                    "channel_attention": attention, "pooling": "stats",
                }))
                number += 1
    pooled = []
    for width in (96, 128, 160):
        for depth in (6, 8):
            for attention in ("eca", "se"):
                for pooling in ("attention", "gem"):
                    pooled.append((width, depth, attention, pooling))
    for width, depth, attention, pooling in pooled[:12]:
        rows.append(make("ARCH", number, f"adaptive_w{width}_d{depth}_{attention}_{pooling}", {
            "model_width": width, "model_depth": depth,
            "channel_attention": attention, "pooling": pooling,
        }))
        number += 1
    for width in (64, 96, 128, 160):
        for pooling in ("stats", "attention"):
            for attention in ("eca", "se"):
                rows.append(make("ARCH", number, f"morphology_w{width}_{attention}_{pooling}", {
                    "backbone": "morphology_resnet", "model_width": width,
                    "model_depth": 6, "channel_attention": attention, "pooling": pooling,
                }))
                number += 1
    for width in (64, 96):
        for depth in (4, 6):
            for kernel in (15, 31):
                rows.append(make("ARCH", number, f"longconv_w{width}_d{depth}_k{kernel}", {
                    "backbone": "longconv", "model_width": width,
                    "model_depth": depth, "model_kernel": kernel, "pooling": "attention",
                }))
                number += 1
    for width in (64, 96):
        for patch in (15, 25):
            for depth in (1, 2):
                rows.append(make("ARCH", number, f"convtransformer_w{width}_p{patch}_d{depth}", {
                    "backbone": "conv_transformer_lite", "model_width": width,
                    "patch_size": patch, "transformer_depth": depth,
                    "attention_heads": 4 if width % 4 == 0 else 2,
                }))
                number += 1
    assert len(rows) == 80
    return rows


def input_configs() -> list[dict]:
    views = (
        "ppg", "ppg_zscore", "ppg_bandpass", "derivatives",
        "derivatives_zscore", "multiview", "multiview_zscore", "multiscale7",
    )
    contexts = (
        (5.0, "none", "5s_none"),
        (10.0, "none", "10s_none"),
        (15.0, "none", "15s_none"),
        (10.0, "raw__fusion+scale", "10s_rawfusion"),
        (10.0, "bandpass_robust__fusion", "10s_robustfusion"),
    )
    rows, number = [], 1
    for view in views:
        for duration, engineered, suffix in contexts:
            rows.append(make("INPUT", number, f"{view}_{suffix}", {
                "view": view, "duration_seconds": duration, "engineered": engineered,
                "model_depth": 7, "pooling": "attention", "augmentation": "jitter_scale",
            }))
            number += 1
    assert len(rows) == 40
    return rows


def objective_variants() -> list[tuple[str, dict]]:
    return [
        ("huber_b025", {"head": "direct", "loss": "huber", "huber_beta": 0.25}),
        ("huber_b050", {"head": "direct", "loss": "huber", "huber_beta": 0.50}),
        ("huber_b100", {"head": "direct", "loss": "huber", "huber_beta": 1.00}),
        ("l1", {"head": "direct", "loss": "l1"}),
        ("logcosh", {"head": "direct", "loss": "logcosh"}),
        ("ccc002", {"head": "direct", "loss": "huber_ccc", "ccc_weight": 0.02}),
        ("ccc005", {"head": "direct", "loss": "huber_ccc", "ccc_weight": 0.05}),
        ("ccc010", {"head": "direct", "loss": "huber_ccc", "ccc_weight": 0.10}),
        ("ccc020", {"head": "direct", "loss": "huber_ccc", "ccc_weight": 0.20}),
        ("asym025", {"head": "direct", "loss": "asymmetric_high", "underestimate_penalty": 0.25}),
        ("asym050", {"head": "direct", "loss": "asymmetric_high", "underestimate_penalty": 0.50}),
        ("focal025", {"head": "direct", "loss": "focal_huber", "focal_weight": 0.25}),
        ("focal050", {"head": "direct", "loss": "focal_huber", "focal_weight": 0.50}),
        ("physio005", {"head": "direct", "loss": "physiology_huber", "physiology_weight": 0.05}),
        ("physio015", {"head": "direct", "loss": "physiology_huber", "physiology_weight": 0.15}),
        ("clinical010", {"head": "clinical", "loss": "huber", "clinical_head_weight": 0.10}),
        ("ordinal010", {"head": "ordinal", "loss": "huber", "ordinal_head_weight": 0.10}),
        ("multitask_light", {"head": "multitask", "loss": "multitask", "aux_weight": 0.20, "consistency_weight": 0.05}),
        ("multitask_default", {"head": "multitask", "loss": "multitask", "aux_weight": 0.35, "consistency_weight": 0.10}),
        ("multitask_strong", {"head": "multitask", "loss": "multitask", "aux_weight": 0.50, "consistency_weight": 0.20}),
    ]


def objective_configs() -> list[dict]:
    contexts = (
        ("derivative5", {"view": "derivatives_zscore", "duration_seconds": 5.0}),
        ("multiview10", {"view": "multiview_zscore", "duration_seconds": 10.0}),
    )
    rows, number = [], 1
    for context_name, context in contexts:
        for label, objective in objective_variants():
            rows.append(make("OBJECTIVE", number, f"{context_name}_{label}", {
                **context, **objective, "model_kernel": 7,
                "pooling": "attention", "augmentation": "jitter_scale",
            }))
            number += 1
    assert len(rows) == 40
    return rows


def training_configs() -> list[dict]:
    optimizers = (
        ("adamw_3e4_cos", {"optimizer": "adamw", "learning_rate": 3e-4, "weight_decay": 2e-4, "scheduler": "cosine"}),
        ("adamw_6e4_none", {"optimizer": "adamw", "learning_rate": 6e-4, "weight_decay": 2e-4, "scheduler": "none"}),
        ("adamw_6e4_cos", {"optimizer": "adamw", "learning_rate": 6e-4, "weight_decay": 2e-4, "scheduler": "cosine"}),
        ("adamw_1e3_cos", {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 2e-4, "scheduler": "cosine"}),
        ("adamw_1e3_wd1e5", {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-5, "scheduler": "none"}),
        ("radam_6e4", {"optimizer": "radam", "learning_rate": 6e-4, "weight_decay": 2e-4, "scheduler": "none"}),
        ("radam_1e3_cos", {"optimizer": "radam", "learning_rate": 1e-3, "weight_decay": 2e-4, "scheduler": "cosine"}),
        ("nadam_6e4", {"optimizer": "nadam", "learning_rate": 6e-4, "weight_decay": 2e-4, "scheduler": "none"}),
        ("nadam_1e3_cos", {"optimizer": "nadam", "learning_rate": 1e-3, "weight_decay": 2e-4, "scheduler": "cosine"}),
        ("sgd_1e2_cos", {"optimizer": "sgd", "learning_rate": 1e-2, "weight_decay": 1e-4, "scheduler": "cosine"}),
    )
    augmentations = ("none", "mask", "amplitude_warp", "compound_morphology")
    rows, number = [], 1
    for optimizer_name, optimizer in optimizers:
        for augmentation in augmentations:
            rows.append(make("TRAIN", number, f"{optimizer_name}_{augmentation}", {
                **optimizer, "augmentation": augmentation, "pooling": "attention",
            }))
            number += 1
    assert len(rows) == 40
    return rows


def all_configs() -> list[dict]:
    rows = architecture_configs() + input_configs() + objective_configs() + training_configs()
    assert len(rows) == 200
    ids = [row["run_id"] for row in rows]
    payloads = [json.dumps({k: v for k, v in row.items() if k not in ("run_id", "innovation_label", "stage")}, sort_keys=True) for row in rows]
    assert len(set(ids)) == 200
    assert len(set(payloads)) == 200
    return rows


def protocol() -> dict:
    configs = all_configs()
    return {
        "version": "1.0.0",
        "locked_before_training": True,
        "created_for": "new 200-method innovation and accuracy search requested for PPG-only SBP/DBP estimation",
        "data": "MIMIC-BP; predefined subject-disjoint train/validation for screening; old test excluded",
        "method_count": 200,
        "families": {
            "ARCH": 80,
            "INPUT": 40,
            "OBJECTIVE": 40,
            "TRAIN": 40,
        },
        "primary_selection": "normalized mean SBP/DBP MAE; clinical tail score and parameter count are tie-breakers",
        "anti_overfitting": "screen all 200 once at seed 314159; confirm only locked finalists using grouped patient CV and fresh seeds",
        "confirmation": "top unique candidates plus prior fixed winner; 5 patient-wise folds x 3 fresh seeds, fixed epoch budget",
        "promotion_gate": "paired patient-level improvement over prior winner with bootstrap CI excluding zero and Holm-adjusted p<0.05",
        "external_gate": "only a confirmed internal winner is evaluated zero-shot on locked VitalDB; destination calibration remains supervised context",
        "claim_limits": [
            "No random segment split",
            "No old MIMIC-BP test use for selection",
            "No clinical-grade/AAMI claim from development data",
            "A screening win alone is not called a confirmed innovation",
        ],
        "literature_anchors": [
            "MIMIC-BP subject-wise benchmark",
            "Generalizable deep learning for PPG-based BP estimation",
            "TransfoRhythm and CNN-Transformer families",
            "2026 mixture-of-experts/time-mixer study",
            "PPG morphology derivatives and physiological multitasking",
            "PaPaGei PPG foundation model",
        ],
        "configs": configs,
    }


def main() -> None:
    payload = protocol()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        existing = json.loads(LOCK.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Refusing to overwrite existing protocol lock: {LOCK}")
    else:
        LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(LOCK)
    print(f"methods={len(payload['configs'])}")


if __name__ == "__main__":
    main()
