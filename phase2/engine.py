from __future__ import annotations

import json
import gc
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from phase2.models import BPModel


ROOT = Path(__file__).resolve().parents[1]
X_PATH = ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy"
META_PATH = ROOT / "outputs/neural/cache/metadata.npz"
FEATURE_PATH = ROOT / "outputs/features/feature_cache.npz"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MemmapDataset(Dataset):
    def __init__(self, indices: np.ndarray, y: np.ndarray, engineered: np.ndarray, augment: str = "none"):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.float32)
        self.engineered = np.asarray(engineered, dtype=np.float32)
        self.augment = augment
        self._x = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self._x is None:
            self._x = np.load(X_PATH, mmap_mode="r", allow_pickle=False)
        x = np.array(self._x[self.indices[item]], dtype=np.float32, copy=True)
        if self.augment == "jitter_scale":
            scale = np.float32(np.random.uniform(0.9, 1.1))
            x *= scale
            x += np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
        elif self.augment == "mask":
            width = np.random.randint(20, 126)
            start = np.random.randint(0, x.shape[-1] - width + 1)
            x[:, start : start + width] = 0
        elif self.augment == "baseline_wander":
            # Low-frequency contact/respiratory drift without label leakage.
            length = x.shape[-1]
            phase = np.float32(np.random.uniform(0, 2 * np.pi))
            cycles = np.float32(np.random.uniform(0.15, 0.8))
            drift = np.float32(np.random.uniform(0.03, 0.18)) * np.sin(
                np.linspace(phase, phase + 2 * np.pi * cycles, length, dtype=np.float32)
            )
            x[0] += drift
        elif self.augment == "motion_spike":
            # Short contact disturbances shared by PPG-derived channels.
            width = int(np.random.randint(4, 35))
            start = int(np.random.randint(0, x.shape[-1] - width + 1))
            pulse = np.hanning(width).astype(np.float32) * np.float32(np.random.uniform(-1.0, 1.0))
            x[:, start : start + width] += pulse[None]
        elif self.augment == "time_shift":
            shift = int(np.random.randint(-62, 63))
            x = np.roll(x, shift, axis=-1).copy()
        elif self.augment == "channel_dropout":
            # Raw PPG remains present; one derivative channel may be missing.
            if x.shape[0] > 1:
                x[int(np.random.randint(1, x.shape[0]))] = 0
        elif self.augment == "compound_clinical":
            x *= np.float32(np.random.uniform(0.92, 1.08))
            x += np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
            if np.random.random() < 0.45:
                length = x.shape[-1]
                drift = np.float32(np.random.uniform(0.02, 0.10)) * np.sin(
                    np.linspace(0, np.random.uniform(0.5, 2.5) * np.pi, length, dtype=np.float32)
                )
                x[0] += drift
            if np.random.random() < 0.25:
                width = int(np.random.randint(8, 64))
                start = int(np.random.randint(0, x.shape[-1] - width + 1))
                x[:, start : start + width] = 0
        elif self.augment == "amplitude_warp":
            length = x.shape[-1]
            knots = np.random.uniform(0.85, 1.15, size=8).astype(np.float32)
            warp = np.interp(np.linspace(0, 7, length), np.arange(8), knots).astype(np.float32)
            x *= warp[None]
        elif self.augment == "colored_noise":
            white = np.random.normal(0, 0.02, size=x.shape[-1]).astype(np.float32)
            colored = np.convolve(white, np.ones(9, np.float32) / 9, mode="same").astype(np.float32)
            x += colored[None]
        elif self.augment == "random_lowpass":
            width = int(np.random.choice([3, 5, 7, 9]))
            kernel = np.ones(width, np.float32) / width
            x = np.stack([np.convolve(channel, kernel, mode="same") for channel in x]).astype(np.float32)
        elif self.augment == "compound_morphology":
            x *= np.float32(np.random.uniform(0.9, 1.1))
            x += np.random.normal(0, 0.012, size=x.shape).astype(np.float32)
            if np.random.random() < 0.5:
                knots = np.random.uniform(0.9, 1.1, size=6).astype(np.float32)
                warp = np.interp(np.linspace(0, 5, x.shape[-1]), np.arange(6), knots).astype(np.float32)
                x *= warp[None]
            if np.random.random() < 0.3:
                shift = int(np.random.randint(-31, 32))
                x = np.roll(x, shift, axis=-1).copy()
        return torch.from_numpy(x), torch.from_numpy(self.engineered[item]), torch.from_numpy(self.y[item])


def load_arrays() -> dict:
    with np.load(META_PATH, allow_pickle=False) as meta, np.load(FEATURE_PATH, allow_pickle=False) as feat:
        result = {k: meta[k].copy() for k in meta.files}
        result["features"] = {k: feat[k].copy() for k in feat.files if "__" in k}
    quality_path = ROOT / "phase2/outputs/clinical_sqi/clinical_sqi.npz"
    if quality_path.exists():
        with np.load(quality_path, allow_pickle=False) as quality:
            result["quality"] = quality["quality_score"].copy()
    return result


def get_engineered(arrays: dict, name: str, train_idx: np.ndarray, eval_idx: np.ndarray):
    if name == "none":
        return np.empty((len(train_idx), 0), np.float32), np.empty((len(eval_idx), 0), np.float32), {}
    scale = name.endswith("+scale")
    key = name.removesuffix("+scale")
    values = arrays["features"][key].astype(np.float32)
    if scale:
        values = np.column_stack([values, arrays["raw_center"], arrays["raw_iqr"]]).astype(np.float32)
    mean = values[train_idx].mean(0)
    std = values[train_idx].std(0).clip(1e-5)
    train = np.clip((values[train_idx] - mean) / std, -10, 10).astype(np.float32)
    evaluate = np.clip((values[eval_idx] - mean) / std, -10, 10).astype(np.float32)
    return train, evaluate, {"mean": mean, "std": std, "key": key, "scale": scale}


def metrics(y: np.ndarray, pred: np.ndarray, train_std: np.ndarray) -> dict:
    error = pred - y
    mae = np.abs(error).mean(0)
    rmse = np.sqrt((error ** 2).mean(0))
    correlation = [float(np.corrcoef(y[:, i], pred[:, i])[0, 1]) for i in range(2)]
    absolute = np.abs(error)
    tail = (y[:, 0] < 95) | (y[:, 0] > 150) | (y[:, 1] < 55) | (y[:, 1] > 95)
    high = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    low = (y[:, 0] < 90) | (y[:, 1] < 60)
    predicted_high = (pred[:, 0] >= 140) | (pred[:, 1] >= 90)
    tp = np.sum(high & predicted_high)
    fn = np.sum(high & ~predicted_high)
    tn = np.sum(~high & ~predicted_high)
    fp = np.sum(~high & predicted_high)
    tail_mae = absolute[tail].mean(0) if np.any(tail) else np.full(2, np.nan)
    low_mae = absolute[low].mean(0) if np.any(low) else np.full(2, np.nan)
    high_mae = absolute[high].mean(0) if np.any(high) else np.full(2, np.nan)
    result = {
        "mae_sbp": float(mae[0]), "mae_dbp": float(mae[1]),
        "mean_mae": float(mae.mean()), "worst_target_mae": float(mae.max()),
        "rmse_sbp": float(rmse[0]), "rmse_dbp": float(rmse[1]),
        "bias_sbp": float(error[:, 0].mean()), "bias_dbp": float(error[:, 1].mean()),
        "pearson_sbp": correlation[0], "pearson_dbp": correlation[1],
        "selection_score": float(np.mean(mae / train_std)),
        "tail_count": int(tail.sum()),
        "tail_mae_sbp": float(tail_mae[0]), "tail_mae_dbp": float(tail_mae[1]),
        "high_count": int(high.sum()),
        "high_mae_sbp": float(high_mae[0]), "high_mae_dbp": float(high_mae[1]),
        "low_count": int(low.sum()),
        "low_mae_sbp": float(low_mae[0]), "low_mae_dbp": float(low_mae[1]),
        "high_bp_sensitivity": float(tp / max(tp + fn, 1)),
        "high_bp_specificity": float(tn / max(tn + fp, 1)),
        "sbp_within_5": float(np.mean(absolute[:, 0] <= 5)),
        "sbp_within_10": float(np.mean(absolute[:, 0] <= 10)),
        "sbp_within_15": float(np.mean(absolute[:, 0] <= 15)),
        "dbp_within_5": float(np.mean(absolute[:, 1] <= 5)),
        "dbp_within_10": float(np.mean(absolute[:, 1] <= 10)),
        "dbp_within_15": float(np.mean(absolute[:, 1] <= 15)),
    }
    # A secondary, explicitly clinical screening score. Overall normalized
    # error stays dominant; tail performance receives a prespecified 25% term.
    result["clinical_selection_score"] = float(
        0.75 * result["selection_score"] + 0.25 * np.mean(tail_mae / train_std)
    )
    return result


def loss_value(config: dict, output: torch.Tensor, target: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    head, name = config.get("head", "direct"), config.get("loss", "huber")
    huber_beta = float(config.get("huber_beta", 0.5))
    if head == "quantile":
        pred = output.view(-1, 2, 3)
        q = torch.tensor([0.1, 0.5, 0.9], device=output.device).view(1, 1, 3)
        delta = target.unsqueeze(-1) - pred
        return torch.maximum(q * delta, (q - 1) * delta).mean()
    prediction = output[:, :2]
    if head == "gaussian":
        log_var = output[:, 2:].clamp(-6, 5)
        return (0.5 * (torch.exp(-log_var) * (prediction - target) ** 2 + log_var)).mean()
    if name == "l1":
        return torch.abs(prediction - target).mean()
    if name == "logcosh":
        residual = (prediction - target).clamp(-10, 10)
        return (residual + nn.functional.softplus(-2 * residual) - math.log(2.0)).mean()
    if name == "huber_ccc":
        base = nn.functional.smooth_l1_loss(prediction, target, beta=huber_beta)
        pred_centered = prediction - prediction.mean(0, keepdim=True)
        target_centered = target - target.mean(0, keepdim=True)
        covariance = (pred_centered * target_centered).mean(0)
        ccc = 2 * covariance / (
            prediction.var(0, unbiased=False) + target.var(0, unbiased=False)
            + (prediction.mean(0) - target.mean(0)).square() + 1e-6
        )
        return base + float(config.get("ccc_weight", 0.1)) * (1 - ccc).mean()
    if name == "asymmetric_high":
        physical = target * target_std + target_mean
        element = nn.functional.smooth_l1_loss(prediction, target, beta=huber_beta, reduction="none")
        # Missing a high BP value is penalized more than overcalling it.
        high = ((physical[:, 0] >= 140) | (physical[:, 1] >= 90)).float().unsqueeze(1)
        under = (prediction < target).float()
        weight = 1 + high * under * float(config.get("underestimate_penalty", 0.75))
        return (element * weight).mean()
    if name == "focal_huber":
        element = nn.functional.smooth_l1_loss(prediction, target, beta=huber_beta, reduction="none")
        difficulty = (prediction.detach() - target).abs().clamp(0, 3)
        return (element * (1 + float(config.get("focal_weight", 0.5)) * difficulty)).mean()
    if name == "physiology_huber":
        base = nn.functional.smooth_l1_loss(prediction, target, beta=huber_beta)
        physical_pred = prediction * target_std + target_mean
        pp = physical_pred[:, 0] - physical_pred[:, 1]
        # Soft constraints only discourage impossible order and extreme pulse
        # pressure; they do not force a narrow normal range.
        penalty = (
            nn.functional.relu(physical_pred[:, 1] - physical_pred[:, 0] + 1).mean()
            + 0.1 * nn.functional.relu(15 - pp).mean()
            + 0.05 * nn.functional.relu(pp - 120).mean()
        ) / 20.0
        return base + float(config.get("physiology_weight", 0.1)) * penalty
    if name == "tail_huber":
        physical = target * target_std + target_mean
        weights = 1 + 1.5 * ((physical[:, 0] < 95) | (physical[:, 0] > 150) | (physical[:, 1] < 55) | (physical[:, 1] > 95)).float()
        element = nn.functional.smooth_l1_loss(prediction, target, beta=huber_beta, reduction="none").mean(1)
        return (element * weights).mean()
    element = nn.functional.smooth_l1_loss(prediction, target, beta=huber_beta, reduction="none")
    target_weights = torch.tensor(config.get("target_weights", [1.0, 1.0]), device=output.device, dtype=element.dtype)
    base = (element * target_weights).mean() / target_weights.mean().clamp_min(1e-6)
    if head == "clinical":
        physical = target * target_std + target_mean
        sbp_class = torch.bucketize(physical[:, 0], torch.tensor([90.0, 120.0, 140.0], device=output.device))
        dbp_class = torch.bucketize(physical[:, 1], torch.tensor([60.0, 80.0, 90.0], device=output.device))
        auxiliary = (
            nn.functional.cross_entropy(output[:, 2:6], sbp_class)
            + nn.functional.cross_entropy(output[:, 6:10], dbp_class)
        ) / 2
        return base + float(config.get("clinical_head_weight", 0.15)) * auxiliary
    if head == "ordinal":
        physical = target * target_std + target_mean
        sbp_thresholds = torch.tensor([90.0, 120.0, 140.0, 160.0], device=output.device)
        dbp_thresholds = torch.tensor([60.0, 80.0, 90.0, 100.0], device=output.device)
        ordinal_target = torch.cat([
            (physical[:, :1] >= sbp_thresholds).float(),
            (physical[:, 1:2] >= dbp_thresholds).float(),
        ], dim=1)
        auxiliary = nn.functional.binary_cross_entropy_with_logits(output[:, 2:10], ordinal_target)
        return base + float(config.get("ordinal_head_weight", 0.15)) * auxiliary
    if head == "multitask":
        physical = target * target_std + target_mean
        map_target = (physical[:, 0] + 2 * physical[:, 1]) / 3
        pp_target = physical[:, 0] - physical[:, 1]
        aux_target = torch.stack([(map_target - 90) / 15, (pp_target - 50) / 15], 1)
        aux = nn.functional.smooth_l1_loss(output[:, 2:], aux_target, beta=huber_beta)
        pred_physical = prediction * target_std + target_mean
        derived = torch.stack([((pred_physical[:, 0] + 2 * pred_physical[:, 1]) / 3 - 90) / 15, (pred_physical[:, 0] - pred_physical[:, 1] - 50) / 15], 1)
        consistency = torch.abs(output[:, 2:] - derived).mean()
        return base + float(config.get("aux_weight", 0.35)) * aux + float(config.get("consistency_weight", 0.1)) * consistency
    return base


@torch.inference_mode()
def predict(model, loader, device, mean, std):
    model.eval()
    outputs, targets = [], []
    for x, engineered, y in loader:
        raw = model(x.to(device, non_blocking=True), engineered.to(device, non_blocking=True))
        outputs.append(model.point_prediction(raw).float().cpu().numpy())
        targets.append(y.numpy())
    pn, yn = np.concatenate(outputs), np.concatenate(targets)
    return yn * std + mean, pn * std + mean


def run_experiment(config: dict, arrays: dict, train_idx: np.ndarray, eval_idx: np.ndarray, out_dir: Path, seed: int = 42, save_model: bool = False) -> dict:
    seed_all(seed)
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    y = arrays["y"].astype(np.float32)
    mean, std = y[train_idx].mean(0), y[train_idx].std(0).clip(1e-5)
    y_train, y_eval = (y[train_idx] - mean) / std, (y[eval_idx] - mean) / std
    e_train, e_eval, e_stats = get_engineered(arrays, config.get("engineered", "none"), train_idx, eval_idx)
    train_ds = MemmapDataset(train_idx, y_train, e_train, config.get("augmentation", "none"))
    eval_ds = MemmapDataset(eval_idx, y_eval, e_eval)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = True
    balance = config.get("balance", "none")
    if balance != "none":
        physical = y[train_idx]
        w = np.ones(len(train_idx), np.float64)
        tail_mask = (physical[:, 0] < 95) | (physical[:, 0] > 150) | (physical[:, 1] < 55) | (physical[:, 1] > 95)
        if balance == "tail":
            w[tail_mask] = float(config.get("tail_sampling_weight", 2.5))
        elif balance in ("sbp_quantile", "clinical_stage"):
            if balance == "sbp_quantile":
                bins = np.quantile(physical[:, 0], np.linspace(0, 1, 9))
                labels = np.clip(np.digitize(physical[:, 0], np.unique(bins)[1:-1]), 0, 7)
            else:
                labels = np.digitize(physical[:, 0], [90, 120, 140, 160]) * 5 + np.digitize(physical[:, 1], [60, 80, 90, 100])
            counts = np.bincount(labels)
            w = 1 / np.maximum(counts[labels], 1)
            w /= np.mean(w)
            w = np.clip(w, 0.25, float(config.get("max_sampling_weight", 4.0)))
        elif balance in ("quality", "quality_tail"):
            if "quality" not in arrays:
                raise FileNotFoundError("Clinical SQI cache is required for quality-aware sampling")
            quality = np.clip(arrays["quality"][train_idx], 0.05, 1.0)
            w = quality.astype(np.float64)
            if balance == "quality_tail":
                w[tail_mask] *= float(config.get("tail_sampling_weight", 2.0))
            w /= np.mean(w)
        else:
            raise ValueError(f"Unknown balance mode: {balance}")
        sampler = WeightedRandomSampler(w, len(w), replacement=True, generator=generator)
        shuffle = False
    # Hundreds of sequential fits can otherwise accumulate multiprocessing
    # file descriptors before Python finalizers run. Non-persistent workers
    # trade a small startup cost for bounded resources and reliable resume.
    workers = min(2, os.cpu_count() or 1)
    kwargs = dict(
        batch_size=int(config.get("batch_size", 192)),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, generator=generator if shuffle else None, **kwargs)
    eval_loader = DataLoader(eval_ds, shuffle=False, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPModel(config, e_train.shape[1]).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = config.get("optimizer", "adamw")
    optimizer_args = {
        "lr": float(config.get("learning_rate", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 2e-4)),
    }
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable, **optimizer_args)
    elif optimizer_name == "radam":
        optimizer = torch.optim.RAdam(trainable, **optimizer_args)
    elif optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(trainable, **optimizer_args)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(trainable, momentum=0.9, nesterov=True, **optimizer_args)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler_name = config.get("scheduler", "none")
    epochs = int(config.get("epochs", 8))
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=optimizer_args["lr"] * 0.05)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.3)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    history, best, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, engineered, target in train_loader:
            x, engineered, target = x.to(device, non_blocking=True), engineered.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(x, engineered)
                loss = loss_value(config, output, target, mean_t, std_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 3.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        true, pred = predict(model, eval_loader, device, mean, std)
        score = metrics(true, pred, std)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **score}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"{config['run_id']} epoch={epoch} loss={row['train_loss']:.4f} MAE={score['mae_sbp']:.3f}/{score['mae_dbp']:.3f} score={score['selection_score']:.4f}", flush=True)
        if config.get("fixed_epochs", False):
            # In an outer confirmation fold the evaluation fold must never
            # choose the epoch. Keep the last epoch dictated by inner/screening
            # validation instead of selecting the best outer-fold score.
            best, stale = score["selection_score"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            np.savez_compressed(out_dir / "best_predictions.npz", indices=eval_idx, y_true=true, y_pred=pred)
        elif score["selection_score"] < best - 1e-5:
            best, stale = score["selection_score"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            np.savez_compressed(out_dir / "best_predictions.npz", indices=eval_idx, y_true=true, y_pred=pred)
        else:
            stale += 1
        if stale >= int(config.get("patience", 3)):
            break
    best_row = history[-1] if config.get("fixed_epochs", False) else min(history, key=lambda z: z["selection_score"])
    best_row.update({
        "run_id": config["run_id"], "stage": config["stage"], "backbone": config["backbone"],
        "view": config["view"], "duration_seconds": config["duration_seconds"], "engineered": config.get("engineered", "none"),
        "head": config.get("head", "direct"), "loss": config.get("loss", "huber"), "augmentation": config.get("augmentation", "none"), "balance": config.get("balance", "none"),
        "seed": seed, "parameters": parameters, "trainable_parameters": sum(p.numel() for p in trainable),
        "train_samples": len(train_idx), "eval_samples": len(eval_idx), "runtime_seconds": time.time() - started,
    })
    if save_model and best_state is not None:
        torch.save({"model_state": best_state, "config": config, "target_mean": mean, "target_std": std, "engineered_stats": e_stats, "metrics": best_row}, out_dir / "best_model.pt")
    (out_dir / "result.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    del model, optimizer, scheduler, best_state, train_loader, eval_loader, train_ds, eval_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_row
