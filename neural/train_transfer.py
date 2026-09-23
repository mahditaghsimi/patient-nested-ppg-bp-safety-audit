from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, seed_everything, write_json
from neural.model import BPTransferTCN


class CachedDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        indices: np.ndarray,
        y_normalized: np.ndarray,
        engineered: np.ndarray,
    ) -> None:
        self.x_path = x_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y_normalized, dtype=np.float32)
        self.engineered = np.asarray(engineered, dtype=np.float32)
        self._x = None

    def _array(self):
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r", allow_pickle=False)
        return self._x

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        global_index = self.indices[item]
        x = np.asarray(self._array()[global_index], dtype=np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.engineered[item]),
            torch.from_numpy(self.y[item]),
        )


def physical_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_std: np.ndarray) -> dict:
    error = y_pred - y_true
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error * error, axis=0))
    bias = np.mean(error, axis=0)
    return {
        "mae_sbp": float(mae[0]),
        "mae_dbp": float(mae[1]),
        "rmse_sbp": float(rmse[0]),
        "rmse_dbp": float(rmse[1]),
        "bias_sbp": float(bias[0]),
        "bias_dbp": float(bias[1]),
        "normalized_mae_score": float(0.5 * (mae[0] / target_std[0] + mae[1] / target_std[1])),
    }


@torch.inference_mode()
def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    predictions, targets = [], []
    for x, engineered, y in loader:
        prediction = model(x.to(device, non_blocking=True), engineered.to(device, non_blocking=True))
        predictions.append(prediction.float().cpu().numpy())
        targets.append(y.numpy())
    pred_n = np.concatenate(predictions)
    y_n = np.concatenate(targets)
    pred = pred_n * target_std + target_mean
    y = y_n * target_std + target_mean
    return physical_metrics(y, pred, target_std), y, pred


def train_mode(
    mode: str,
    cfg: dict,
    arrays: dict,
    output_dir: Path,
    device: torch.device,
    train_fraction: float = 1.0,
    epochs_override: int | None = None,
    run_name: str | None = None,
) -> dict:
    seed_everything(int(cfg["seed"]))
    hybrid = mode == "finetuned_hybrid"
    pretrained = mode != "scratch"
    frozen = mode == "frozen"
    split = arrays["split"]
    train_idx = np.flatnonzero(split == "train")
    if train_fraction < 1.0:
        rng = np.random.default_rng(int(cfg["seed"]))
        train_patients = np.unique(arrays["patient"][train_idx])
        count = max(1, int(np.ceil(train_fraction * len(train_patients))))
        selected_patients = rng.choice(train_patients, size=count, replace=False)
        train_idx = train_idx[np.isin(arrays["patient"][train_idx], selected_patients)]
    val_idx = np.flatnonzero(split == "validation")
    target_mean = arrays["y"][train_idx].mean(0).astype(np.float32)
    target_std = arrays["y"][train_idx].std(0).astype(np.float32)
    y_train = (arrays["y"][train_idx] - target_mean) / target_std
    y_val = (arrays["y"][val_idx] - target_mean) / target_std

    engineered_all = arrays["engineered"]
    engineered_mean = engineered_all[train_idx].mean(0).astype(np.float32)
    engineered_std = engineered_all[train_idx].std(0).astype(np.float32)
    engineered_std = np.maximum(engineered_std, 1e-5)
    if hybrid:
        e_train = np.clip((engineered_all[train_idx] - engineered_mean) / engineered_std, -10, 10)
        e_val = np.clip((engineered_all[val_idx] - engineered_mean) / engineered_std, -10, 10)
    else:
        e_train = np.empty((len(train_idx), 0), dtype=np.float32)
        e_val = np.empty((len(val_idx), 0), dtype=np.float32)

    x_path = Path("outputs/neural/cache/ppg15s_3ch_float16.npy")
    train_ds = CachedDataset(x_path, train_idx, y_train, e_train)
    val_ds = CachedDataset(x_path, val_idx, y_val, e_val)
    kwargs = dict(
        batch_size=int(cfg["neural"]["batch_size"]),
        num_workers=int(cfg["neural"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["neural"]["num_workers"]) > 0,
    )
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, drop_last=False, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)

    model = BPTransferTCN(engineered_dim=e_train.shape[1]).to(device)
    if pretrained:
        checkpoint = torch.load(cfg["forecast_checkpoint"], map_location="cpu", weights_only=False)
        model.load_forecasting_encoder(checkpoint)
    if frozen:
        model.freeze_encoder()
    encoder_parameters = list(model.stem.parameters()) + list(model.blocks.parameters())
    head_parameters = list(model.bp_head.parameters())
    parameter_groups = [{"params": head_parameters, "lr": float(cfg["neural"]["learning_rate_head"])}]
    if not frozen:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": float(cfg["neural"]["learning_rate_encoder"])}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(cfg["neural"]["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.SmoothL1Loss(beta=0.5)
    mode_dir = output_dir / (run_name or mode)
    mode_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    epochs = int(epochs_override or cfg["neural"]["epochs_screen"])
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        if frozen:
            model.stem.eval()
            model.blocks.eval()
        losses = []
        for x, engineered, y in train_loader:
            x = x.to(device, non_blocking=True)
            engineered = engineered.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(x, engineered)
                loss = criterion(prediction, y) + 0.05 * torch.mean(torch.abs(prediction - y))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_metrics, y_true, y_pred = evaluate(model, val_loader, device, target_mean, target_std)
        row = {
            "mode": mode,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **val_metrics,
            "runtime_seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(mode_dir / "history.csv", index=False)
        score = val_metrics["normalized_mae_score"]
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": mode,
                    "engineered_dim": e_train.shape[1],
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "engineered_mean": engineered_mean,
                    "engineered_std": engineered_std,
                    "best_epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                mode_dir / "best_model.pt",
            )
            np.savez_compressed(mode_dir / "validation_predictions.npz", y_true=y_true, y_pred=y_pred)
        else:
            stale += 1
        print(
            f"{mode} epoch {epoch}/{epochs}: loss={row['train_loss']:.4f}, "
            f"MAE={row['mae_sbp']:.3f}/{row['mae_dbp']:.3f}, score={score:.4f}",
            flush=True,
        )
        if stale >= int(cfg["neural"]["patience"]):
            break
    best_row = min(history, key=lambda item: item["normalized_mae_score"])
    best_row["best_epoch"] = best_epoch
    best_row["train_fraction"] = float(train_fraction)
    best_row["train_patients"] = int(len(np.unique(arrays["patient"][train_idx])))
    best_row["train_samples"] = int(len(train_idx))
    best_row["parameters"] = sum(p.numel() for p in model.parameters())
    best_row["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return best_row


def main() -> None:
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_path = Path("outputs/neural/cache/metadata.npz")
    x_path = Path("outputs/neural/cache/ppg15s_3ch_float16.npy")
    if not metadata_path.exists() or not x_path.exists():
        raise FileNotFoundError("Run neural/build_input_cache.py first")
    with np.load(metadata_path, allow_pickle=False) as meta, np.load(
        cfg["features"]["cache_path"], allow_pickle=False
    ) as features:
        arrays = {
            "split": meta["split"].copy(),
            "y": meta["y"].copy(),
            "patient": meta["patient_id"].copy(),
            "engineered": features["raw__fusion"].copy(),
        }
        if not np.array_equal(meta["patient_id"], features["patient_id"]):
            raise RuntimeError("Neural and mathematical feature caches are misaligned")
    output_dir = Path("outputs/neural/transfer_screen")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("scratch", "frozen", "finetuned", "finetuned_hybrid"):
        rows.append(train_mode(mode, cfg, arrays, output_dir, device))
        pd.DataFrame(rows).sort_values("normalized_mae_score").to_csv(
            output_dir / "comparison.csv", index=False
        )
    comparison = pd.DataFrame(rows).sort_values("normalized_mae_score")
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    write_json(output_dir / "best_neural.json", comparison.iloc[0].to_dict())
    print("\n", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
