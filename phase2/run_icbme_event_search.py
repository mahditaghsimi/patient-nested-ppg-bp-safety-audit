from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.models import BACKBONES, InputView


CACHE = ROOT / "outputs/neural/cache/ppg15s_3ch_float16.npy"
META = ROOT / "outputs/neural/cache/metadata.npz"
OUT = ROOT / "phase2/outputs/icbme_event_search"
LOCK = ROOT / "phase2/protocol/ICBME_EVENT_SEARCH_LOCK.json"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EventDataset(Dataset):
    def __init__(self, indices: np.ndarray, events: np.ndarray, y_normalized: np.ndarray, augmentation: str):
        self.indices = np.asarray(indices, np.int64)
        self.events = np.asarray(events, np.float32)
        self.y = np.asarray(y_normalized, np.float32)
        self.augmentation = augmentation
        self.x = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        if self.x is None:
            self.x = np.load(CACHE, mmap_mode="r", allow_pickle=False)
        signal = np.asarray(self.x[self.indices[item]], dtype=np.float32).copy()
        if self.augmentation in ("jitter", "compound"):
            signal *= np.float32(np.random.uniform(0.92, 1.08))
            signal += np.random.normal(0, 0.01, signal.shape).astype(np.float32)
        if self.augmentation in ("mask", "compound") and np.random.random() < 0.4:
            width = int(np.random.randint(12, 100))
            start = int(np.random.randint(0, signal.shape[-1] - width + 1))
            signal[:, start : start + width] = 0
        if self.augmentation == "time_shift":
            signal = np.roll(signal, int(np.random.randint(-62, 63)), axis=-1).copy()
        return torch.from_numpy(signal), torch.from_numpy(self.events[item]), torch.from_numpy(self.y[item])


class EventModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.view = InputView(config["view"], config["duration_seconds"])
        self.backbone = BACKBONES[config["backbone"]](self.view.out_channels)
        hidden = int(config.get("hidden", 192))
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(float(config.get("dropout", 0.15))), nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 4),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.view(signal)))


def make_config(number: int, label: str, changes: dict) -> dict:
    config = {
        "backbone": "resnet", "view": "multiview", "duration_seconds": 10.0,
        "objective": "bce", "pos_weight": "sqrt", "augmentation": "none", "sampling": "none",
        "joint_regression_weight": 0.0, "focal_gamma": 2.0, "hidden": 192, "dropout": 0.15,
        "epochs": 6, "patience": 2, "batch_size": 192, "learning_rate": 1e-3, "weight_decay": 2e-4,
    }
    config.update(changes)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    config["label"] = label
    config["run_id"] = f"EVENT_{number:03d}_{label}_{digest}"
    return config


def configs() -> list[dict]:
    rows, number = [], 1
    representations = [
        ("resnet_ppg5", {"backbone": "resnet", "view": "ppg_zscore", "duration_seconds": 5.0}),
        ("resnet_deriv5", {"backbone": "resnet", "view": "derivatives", "duration_seconds": 5.0}),
        ("resnet_multi10", {"backbone": "resnet", "view": "multiview", "duration_seconds": 10.0}),
        ("resnet_deriv15", {"backbone": "resnet", "view": "derivatives", "duration_seconds": 15.0}),
        ("gated_multi10", {"backbone": "gated_tcn", "view": "multiview", "duration_seconds": 10.0}),
        ("inception_deriv10", {"backbone": "inception", "view": "derivatives", "duration_seconds": 10.0}),
        ("spectral_deriv10", {"backbone": "spectral_fusion", "view": "derivatives", "duration_seconds": 10.0}),
        ("patch_deriv10", {"backbone": "patch_transformer", "view": "derivatives", "duration_seconds": 10.0}),
    ]
    for label, change in representations:
        rows.append(make_config(number, label, change)); number += 1
    objectives = [
        ("bce_none", {"objective": "bce", "pos_weight": "none"}),
        ("bce_sqrt", {"objective": "bce", "pos_weight": "sqrt"}),
        ("bce_full", {"objective": "bce", "pos_weight": "full"}),
        ("focal1", {"objective": "focal", "pos_weight": "sqrt", "focal_gamma": 1.0}),
        ("focal2", {"objective": "focal", "pos_weight": "sqrt", "focal_gamma": 2.0}),
        ("focal3", {"objective": "focal", "pos_weight": "sqrt", "focal_gamma": 3.0}),
        ("joint10", {"objective": "bce", "pos_weight": "sqrt", "joint_regression_weight": 0.10}),
        ("joint25", {"objective": "bce", "pos_weight": "sqrt", "joint_regression_weight": 0.25}),
        ("joint50", {"objective": "bce", "pos_weight": "sqrt", "joint_regression_weight": 0.50}),
        ("sample_combo", {"objective": "bce", "pos_weight": "none", "sampling": "event_combo"}),
    ]
    for base_label, base in [("multi10", representations[2][1]), ("deriv5", representations[1][1])]:
        for label, change in objectives:
            rows.append(make_config(number, f"{base_label}_{label}", {**base, **change})); number += 1
    robust = [
        ("aug_jitter", {"augmentation": "jitter"}), ("aug_mask", {"augmentation": "mask"}),
        ("aug_shift", {"augmentation": "time_shift"}), ("aug_compound", {"augmentation": "compound"}),
        ("drop05", {"dropout": 0.05}), ("drop30", {"dropout": 0.30}),
        ("lr5e4", {"learning_rate": 5e-4}), ("wd1e3", {"weight_decay": 1e-3}),
    ]
    for label, change in robust:
        rows.append(make_config(number, label, {**representations[2][1], **change})); number += 1
    return rows


def protocol() -> dict:
    rows = configs()
    return {
        "version": "1.0.0", "locked_before_training": True,
        "tasks": {"high_bp": "SBP>=140 or DBP>=90", "hypotension": "derived MAP<65"},
        "selection": "0.65 high-BP average precision + 0.35 hypotension average precision",
        "split": "predefined patient-disjoint train/validation; old test excluded",
        "threshold_reporting": ["0.5", "sensitivity at >=90% specificity"],
        "config_count": len(rows), "configs": rows,
    }


def event_metrics(events: np.ndarray, probability: np.ndarray) -> dict:
    result = {}
    for column, name in enumerate(["high", "map_low"]):
        truth, score = events[:, column].astype(bool), probability[:, column]
        fpr, tpr, thresholds = roc_curve(truth, score)
        eligible = np.flatnonzero(fpr <= 0.10)
        index = eligible[np.argmax(tpr[eligible])] if len(eligible) else 0
        prediction = score >= 0.5
        result.update({
            f"{name}_prevalence": float(truth.mean()),
            f"{name}_auroc": float(roc_auc_score(truth, score)),
            f"{name}_auprc": float(average_precision_score(truth, score)),
            f"{name}_brier": float(brier_score_loss(truth, score)),
            f"{name}_sensitivity_05": float(np.sum(truth & prediction) / max(np.sum(truth), 1)),
            f"{name}_specificity_05": float(np.sum(~truth & ~prediction) / max(np.sum(~truth), 1)),
            f"{name}_sensitivity_at_90spec": float(tpr[index]),
            f"{name}_threshold_at_90spec": float(thresholds[index]),
        })
    result["selection_score"] = 0.65 * result["high_auprc"] + 0.35 * result["map_low_auprc"]
    return result


def loss_value(config: dict, output: torch.Tensor, event: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    logits, regression = output[:, :2], output[:, 2:]
    element = nn.functional.binary_cross_entropy_with_logits(logits, event, pos_weight=pos_weight, reduction="none")
    if config["objective"] == "focal":
        probability = torch.sigmoid(logits)
        pt = event * probability + (1 - event) * (1 - probability)
        element = element * (1 - pt).pow(float(config["focal_gamma"]))
    loss = element.mean()
    if config["joint_regression_weight"]:
        loss = loss + float(config["joint_regression_weight"]) * nn.functional.smooth_l1_loss(regression, y, beta=0.5)
    return loss


@torch.inference_mode()
def evaluate(model: EventModel, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval(); truth, probability = [], []
    for signal, event, _ in loader:
        output = model(signal.to(device, non_blocking=True))
        probability.append(torch.sigmoid(output[:, :2]).float().cpu().numpy())
        truth.append(event.numpy())
    truth, probability = np.concatenate(truth), np.concatenate(probability)
    return event_metrics(truth, probability), truth, probability


def run(config: dict, train: np.ndarray, validation: np.ndarray, y: np.ndarray, seed: int, smoke: bool) -> dict:
    seed_all(seed); device = torch.device("cuda")
    mean, std = y[train].mean(0), y[train].std(0).clip(1e-5)
    yn = (y - mean) / std
    map_value = (y[:, 0] + 2 * y[:, 1]) / 3
    events = np.column_stack([(y[:, 0] >= 140) | (y[:, 1] >= 90), map_value < 65]).astype(np.float32)
    train_ds = EventDataset(train, events[train], yn[train], config["augmentation"])
    val_ds = EventDataset(validation, events[validation], yn[validation], "none")
    sampler, shuffle = None, True
    if config["sampling"] == "event_combo":
        combo = events[train, 0].astype(int) * 2 + events[train, 1].astype(int)
        count = np.bincount(combo, minlength=4)
        weights = 1 / np.maximum(count[combo], 1)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(seed))
        shuffle = False
    workers = 2 if smoke else 4
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=shuffle, sampler=sampler, num_workers=workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    model = EventModel(config).to(device)
    prevalence = events[train].mean(0).clip(1e-4, 1 - 1e-4)
    ratio = (1 - prevalence) / prevalence
    if config["pos_weight"] == "none": ratio[:] = 1
    elif config["pos_weight"] == "sqrt": ratio = np.sqrt(ratio)
    pos_weight = torch.tensor(ratio, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, best_data, stale, history = -np.inf, None, None, 0, []
    epochs = 1 if smoke else config["epochs"]
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for signal, event, target in train_loader:
            signal, event, target = signal.to(device, non_blocking=True), event.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_value(config, model(signal), event, target, pos_weight)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        current, truth, probability = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **current})
        print(f"{config['run_id']} epoch={epoch} AP_high={current['high_auprc']:.4f} AP_MAPlow={current['map_low_auprc']:.4f} score={current['selection_score']:.4f}", flush=True)
        if current["selection_score"] > best + 1e-5:
            best, stale = current["selection_score"], 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_data = (truth, probability)
        else:
            stale += 1
        if stale >= config["patience"]: break
    best_row = max(history, key=lambda row: row["selection_score"])
    run_dir = OUT / "runs" / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    np.savez_compressed(run_dir / "best_predictions.npz", indices=validation, y_true=best_data[0], y_probability=best_data[1])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return {**best_row, **config, "train_samples": len(train), "validation_samples": len(validation), "status": "ok"}


def journal(event: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not LOCK.exists():
        LOCK.write_text(json.dumps(protocol(), ensure_ascii=False, indent=2), encoding="utf-8")
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    rows_config = locked["configs"][:6] if args.smoke else locked["configs"]
    with np.load(META, allow_pickle=False) as meta:
        split, y = meta["split"].astype(str), meta["y"].astype(np.float32)
    train, validation = np.flatnonzero(split == "train"), np.flatnonzero(split == "validation")
    if args.smoke: train, validation = train[:768], validation[:384]
    output = OUT / ("smoke_results.csv" if args.smoke else "screening_results.csv")
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    completed = {row["run_id"] for row in rows}
    journal("event_search_started", smoke=args.smoke, configs=len(rows_config))
    for number, config in enumerate(rows_config, 1):
        if config["run_id"] in completed: continue
        print(f"[{number}/{len(rows_config)}] {config['run_id']}", flush=True)
        result = run(config, train, validation, y, args.seed, args.smoke)
        rows.append(result); pd.DataFrame(rows).to_csv(output, index=False); journal("event_run_completed", result=result)
    ranking = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    ranking.to_csv(OUT / ("smoke_ranking.csv" if args.smoke else "screening_ranking.csv"), index=False)
    print(ranking[["run_id", "high_auprc", "high_auroc", "high_sensitivity_at_90spec", "map_low_auprc", "map_low_auroc", "selection_score"]].head(15).to_string(index=False))


if __name__ == "__main__": main()
