from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2.models import BPModel


SOURCE_CACHE = ROOT / "phase2/outputs/mimic3_aux_cache/MIMIC3_AUX_15S_3CH_FLOAT16.npy"
SOURCE_META = ROOT / "phase2/outputs/mimic3_aux_cache/AUX_METADATA.csv"
TARGET_CACHE = ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy"
TARGET_META = ROOT / "outputs/neural/cache/metadata.npz"
PAPAGEI = ROOT / "data/pretraining/papagei_s.pt"
OUT = ROOT / "phase2/outputs/mimic3_aux_transfer_screen"


CONFIGS = {
    "resnet_multiview": {"backbone": "resnet", "view": "multiview"},
    "gated_tcn": {"backbone": "gated_tcn", "view": "derivatives"},
    "patch_transformer": {"backbone": "patch_transformer", "view": "derivatives"},
    "spectral_fusion": {"backbone": "spectral_fusion", "view": "derivatives"},
    "papagei_frozen": {
        "backbone": "papagei",
        "view": "ppg_bandpass",
        "foundation_mode": "frozen",
        "papagei_weights": str(PAPAGEI),
    },
    "papagei_last_blocks": {
        "backbone": "papagei",
        "view": "ppg_bandpass",
        "foundation_mode": "last_blocks",
        "papagei_weights": str(PAPAGEI),
    },
}


class WaveDataset(Dataset):
    def __init__(self, path: Path, indices: np.ndarray, y: np.ndarray, augment: bool = False):
        self.path = path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.float32)
        self.augment = augment
        self.array = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        signal = np.array(self.array[self.indices[item]], dtype=np.float32, copy=True)
        if self.augment:
            signal *= np.float32(np.random.uniform(0.92, 1.08))
            signal += np.random.normal(0, 0.01, signal.shape).astype(np.float32)
            if np.random.random() < 0.25:
                width = np.random.randint(12, 80)
                start = np.random.randint(0, signal.shape[-1] - width)
                signal[:, start : start + width] = 0
        return torch.from_numpy(signal), torch.from_numpy(self.y[item])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config(name: str) -> dict:
    value = {
        **CONFIGS[name],
        "duration_seconds": 10.0,
        "head": "direct",
        "dropout": 0.15,
        "head_hidden": 192,
    }
    return value


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - y
    mae = np.abs(error).mean(0)
    rmse = np.sqrt((error**2).mean(0))
    return {
        "sbp_mae": float(mae[0]),
        "dbp_mae": float(mae[1]),
        "mean_mae": float(mae.mean()),
        "sbp_rmse": float(rmse[0]),
        "dbp_rmse": float(rmse[1]),
        "sbp_bias": float(error[:, 0].mean()),
        "dbp_bias": float(error[:, 1].mean()),
        "sbp_r": float(np.corrcoef(y[:, 0], prediction[:, 0])[0, 1]),
        "dbp_r": float(np.corrcoef(y[:, 1], prediction[:, 1])[0, 1]),
    }


@torch.inference_mode()
def evaluate(model: BPModel, loader: DataLoader, device: torch.device, mean: np.ndarray, std: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    truth, pred = [], []
    for signal, target in loader:
        output = model(signal.to(device, non_blocking=True))[:, :2]
        pred.append(output.float().cpu().numpy())
        truth.append(target.numpy())
    truth_n, pred_n = np.concatenate(truth), np.concatenate(pred)
    truth_value = truth_n * std + mean
    pred_value = pred_n * std + mean
    return metrics(truth_value, pred_value), truth_value, pred_value


def fit(
    model: BPModel,
    cache: Path,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    y_physical: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    backbone_lr_factor: float = 1.0,
) -> tuple[BPModel, dict]:
    mean = y_physical[train_idx].mean(0).astype(np.float32)
    std = y_physical[train_idx].std(0).clip(1e-5).astype(np.float32)
    train_y = (y_physical[train_idx] - mean) / std
    eval_y = (y_physical[eval_idx] - mean) / std
    train_loader = DataLoader(
        WaveDataset(cache, train_idx, train_y, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(seed),
    )
    eval_loader = DataLoader(
        WaveDataset(cache, eval_idx, eval_y),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    model.to(device)
    if backbone_lr_factor == 1.0:
        parameter_groups = [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": learning_rate}]
    else:
        backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        other_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids]
        parameter_groups = [
            {"params": backbone_parameters, "lr": learning_rate * backbone_lr_factor},
            {"params": other_parameters, "lr": learning_rate},
        ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state, best_metrics = None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for signal, target in train_loader:
            signal = signal.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(signal)[:, :2]
                loss = nn.functional.smooth_l1_loss(output, target, beta=0.5)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        current, _, _ = evaluate(model, eval_loader, device, mean, std)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **current})
        if best_metrics is None or current["mean_mae"] < best_metrics["mean_mae"]:
            best_metrics = current
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    best_metrics["history"] = history
    return model, best_metrics


def reset_head(model: BPModel) -> None:
    for module in model.head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def append_journal(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), **payload}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--models", nargs="*", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--source-policy", choices=["global_nonoverlap", "fold_safe"], default="global_nonoverlap")
    parser.add_argument(
        "--transfer-strategy",
        choices=["reset_head", "full_model", "frozen_backbone", "discriminative"],
        default="reset_head",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for this stage.")
    if not SOURCE_CACHE.exists() or not SOURCE_META.exists():
        raise FileNotFoundError("Build the clean views and auxiliary cache first.")
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda")

    source = pd.read_csv(SOURCE_META, dtype={"official_subject_id": str})
    with np.load(TARGET_META, allow_pickle=False) as meta:
        target_y = meta["y"].astype(np.float32)
        target_split = meta["split"].astype(str)
        target_patient = meta["patient_id"].astype(str)
    if args.source_policy == "global_nonoverlap":
        safe = source.eligible_for_confirmatory_merge.to_numpy(dtype=bool)
    else:
        held_out_subjects = {value.lstrip("p") for value in target_patient[np.isin(target_split, ["validation", "test"])]}
        safe = (
            source.view_high_confidence.to_numpy(dtype=bool)
            & (source.mapping_status.to_numpy() == "official_unique")
            & ~source.official_subject_id.isin(held_out_subjects).to_numpy()
        )
    if safe.sum() < 1000:
        raise RuntimeError(f"Only {safe.sum()} confirmatory mapped-nonoverlap source windows; refusing silent fallback.")
    source_indices = np.flatnonzero(safe)
    source_y = source[["sbp", "dbp"]].to_numpy(dtype=np.float32)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    source_train_rel, source_val_rel = next(splitter.split(source_indices, groups=source.loc[safe, "group_key"]))
    source_train = source_indices[source_train_rel]
    source_val = source_indices[source_val_rel]

    target_train = np.flatnonzero(target_split == "train")
    target_val = np.flatnonzero(target_split == "validation")
    if args.smoke:
        source_train, source_val = source_train[:512], source_val[:256]
        target_train, target_val = target_train[:512], target_val[:256]
        source_epochs = target_epochs = 1
    else:
        source_epochs, target_epochs = 4, 6

    rows = []
    suffix = f"{args.source_policy}_{args.transfer_strategy}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    results_path = OUT / f"results_{suffix}.csv"
    for model_name in args.models:
        append_journal({"event": "model_started", "model": model_name, "smoke": args.smoke})
        seed_all(args.seed)
        source_model = BPModel(config(model_name))
        source_model, source_result = fit(
            source_model,
            SOURCE_CACHE,
            source_train,
            source_val,
            source_y,
            device,
            source_epochs,
            256,
            args.seed,
            1e-3,
        )
        pretrained_state = copy.deepcopy(source_model.state_dict())
        pretrained_backbone = copy.deepcopy(source_model.backbone.state_dict())
        del source_model
        torch.cuda.empty_cache()

        transfer_mode = f"supervised_aux_{args.transfer_strategy}"
        for mode in ("scratch", transfer_mode):
            seed_all(args.seed)
            target_model = BPModel(config(model_name))
            backbone_lr_factor = 1.0
            if mode != "scratch":
                if args.transfer_strategy == "full_model":
                    target_model.load_state_dict(pretrained_state, strict=True)
                else:
                    target_model.backbone.load_state_dict(pretrained_backbone, strict=True)
                    reset_head(target_model)
                if args.transfer_strategy == "frozen_backbone":
                    target_model.backbone.requires_grad_(False)
                elif args.transfer_strategy == "discriminative":
                    backbone_lr_factor = 0.1
            target_lr = 1e-3 if mode == "scratch" or args.transfer_strategy in {"frozen_backbone", "discriminative"} else 4e-4
            target_model, result = fit(
                target_model,
                TARGET_CACHE,
                target_train,
                target_val,
                target_y,
                device,
                target_epochs,
                192,
                args.seed,
                target_lr,
                backbone_lr_factor,
            )
            row = {
                "model": model_name,
                "mode": mode,
                "seed": args.seed,
                "source_train_windows": int(len(source_train)),
                "source_val_windows": int(len(source_val)),
                "target_train_windows": int(len(target_train)),
                "target_val_windows": int(len(target_val)),
                "source_val_sbp_mae": source_result["sbp_mae"],
                "source_val_dbp_mae": source_result["dbp_mae"],
                **{key: value for key, value in result.items() if key != "history"},
            }
            rows.append(row)
            run_dir = OUT / f"{model_name}_{mode}_{args.source_policy}_seed{args.seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "history.json").write_text(json.dumps(result["history"], indent=2) + "\n", encoding="utf-8")
            torch.save(target_model.state_dict(), run_dir / "best_model.pt")
            append_journal({"event": "run_finished", **row})
            pd.DataFrame(rows).to_csv(results_path, index=False)
            del target_model
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    pivot = results.pivot(index="model", columns="mode", values=["sbp_mae", "dbp_mae", "mean_mae"])
    comparisons = []
    for name in args.models:
        scratch = results[(results.model == name) & (results["mode"] == "scratch")].iloc[0]
        transfer = results[(results.model == name) & (results["mode"] == transfer_mode)].iloc[0]
        comparisons.append(
            {
                "model": name,
                "scratch_sbp_mae": scratch.sbp_mae,
                "pretrained_sbp_mae": transfer.sbp_mae,
                "delta_sbp_mae": transfer.sbp_mae - scratch.sbp_mae,
                "scratch_dbp_mae": scratch.dbp_mae,
                "pretrained_dbp_mae": transfer.dbp_mae,
                "delta_dbp_mae": transfer.dbp_mae - scratch.dbp_mae,
                "delta_mean_mae": transfer.mean_mae - scratch.mean_mae,
            }
        )
    comparison = pd.DataFrame(comparisons).sort_values("delta_mean_mae")
    comparison.to_csv(OUT / f"pretraining_comparison_{suffix}.csv", index=False)
    report = f"""# غربالگری انتقال از MIMIC-III به MIMIC-BP

- دادهٔ کمکی: فقط high-confidence، دارای subject رسمی و non-overlap قطعی؛
- source policy: `{args.source_policy}`؛
- transfer strategy: `{args.transfer_strategy}`؛
- مقصد انتخاب: train/validation رسمی MIMIC-BP؛ test استفاده نشد؛
- seed: {args.seed}؛ smoke: {args.smoke}.

مقدار منفی `delta_*_mae` یعنی pretraining بهتر از scratch بوده است.

```
{comparison.to_string(index=False)}
```

این مرحله screening تک-seed است. فقط برنده باید با چند seed و grouped CV تأیید شود.
"""
    (OUT / f"AUX_TRANSFER_SCREEN_{suffix}_FA.md").write_text(report, encoding="utf-8")
    print(pivot)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
