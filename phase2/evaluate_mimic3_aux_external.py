from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.models import BPModel


RUNS = ROOT / "phase2/outputs/mimic3_aux_transfer_screen"
EXTERNAL = ROOT / "data/external/prepared/ppg_ambulatory_56.npz"
TARGET_META = ROOT / "outputs/neural/cache/metadata.npz"
OUT = RUNS / "external_ambulatory56"
SEEDS = [42, 123, 2026]


class Signals(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return torch.from_numpy(np.asarray(self.x[index], dtype=np.float32))


def config() -> dict:
    return {
        "backbone": "resnet",
        "view": "multiview",
        "duration_seconds": 10.0,
        "head": "direct",
        "dropout": 0.15,
        "head_hidden": 192,
    }


def metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - y
    absolute = np.abs(error)
    return {
        "sbp_mae": float(absolute[:, 0].mean()),
        "dbp_mae": float(absolute[:, 1].mean()),
        "mean_mae": float(absolute.mean()),
        "sbp_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "dbp_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "sbp_bias": float(error[:, 0].mean()),
        "dbp_bias": float(error[:, 1].mean()),
        "sbp_r": float(np.corrcoef(y[:, 0], prediction[:, 0])[0, 1]),
        "dbp_r": float(np.corrcoef(y[:, 1], prediction[:, 1])[0, 1]),
    }


@torch.inference_mode()
def predict(path: Path, loader: DataLoader, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    model = BPModel(config()).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    values = []
    for signal in loader:
        normalized = model(signal.to(device, non_blocking=True))[:, :2]
        values.append(normalized.float().cpu().numpy() * std + mean)
    return np.concatenate(values)


def subject_average(patient: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = np.unique(patient)
    y_subject = np.asarray([y[patient == subject].mean(0) for subject in subjects])
    p_subject = np.asarray([prediction[patient == subject].mean(0) for subject in subjects])
    return subjects, y_subject, p_subject


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(TARGET_META, allow_pickle=False) as target:
        train = target["split"].astype(str) == "train"
        mean = target["y"][train].mean(0).astype(np.float32)
        std = target["y"][train].std(0).clip(1e-5).astype(np.float32)
    with np.load(EXTERNAL, allow_pickle=False) as external:
        x = external["x"].copy()
        y = external["y"].copy()
        patient = external["patient_id"].copy()
    if x.shape[-1] < 1250:
        raise RuntimeError("The 10-second model cannot be evaluated on shorter external signals without invalid padding.")
    loader = DataLoader(Signals(x), batch_size=256, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    modes = {
        "scratch": "resnet_multiview_scratch_fold_safe_seed{seed}/best_model.pt",
        "supervised_aux_reset_head": "resnet_multiview_supervised_aux_pretrain_fold_safe_seed{seed}/best_model.pt",
    }
    rows = []
    all_predictions = {}
    for mode, template in modes.items():
        predictions = []
        for seed in SEEDS:
            path = RUNS / template.format(seed=seed)
            if not path.exists():
                raise FileNotFoundError(path)
            prediction = predict(path, loader, mean, std, device)
            predictions.append(prediction)
            _, y_subject, p_subject = subject_average(patient, y, prediction)
            rows.append({"mode": mode, "member": f"seed{seed}", "subjects": len(y_subject), **metric(y_subject, p_subject)})
        ensemble = np.mean(predictions, axis=0)
        subjects, y_subject, p_subject = subject_average(patient, y, ensemble)
        all_predictions[mode] = p_subject
        rows.append({"mode": mode, "member": "three_seed_ensemble", "subjects": len(subjects), **metric(y_subject, p_subject)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "EXTERNAL_METRICS.csv", index=False)
    np.savez_compressed(
        OUT / "EXTERNAL_PREDICTIONS.npz",
        patient_id=subjects,
        y_true=y_subject,
        scratch=all_predictions["scratch"],
        supervised_aux_reset_head=all_predictions["supervised_aux_reset_head"],
    )
    ensemble = table[table.member == "three_seed_ensemble"].set_index("mode")
    scratch = ensemble.loc["scratch"]
    transfer = ensemble.loc["supervised_aux_reset_head"]
    decision = {
        "dataset": "ambulatory56",
        "evaluation_unit": "subject",
        "subjects": int(scratch.subjects),
        "scratch_sbp_mae": scratch.sbp_mae,
        "pretrained_sbp_mae": transfer.sbp_mae,
        "delta_sbp_mae": transfer.sbp_mae - scratch.sbp_mae,
        "scratch_dbp_mae": scratch.dbp_mae,
        "pretrained_dbp_mae": transfer.dbp_mae,
        "delta_dbp_mae": transfer.dbp_mae - scratch.dbp_mae,
        "delta_mean_mae": transfer.mean_mae - scratch.mean_mae,
        "caveat": "External cuff labels are not simultaneous with waveform windows; this is a domain stress test, not clinical agreement.",
    }
    (OUT / "EXTERNAL_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# ارزیابی خارجی اثر pretraining کمکی

مدل: ensemble سه seed از ResNet multiview؛ واحد ارزیابی: بیمار؛ دیتاست: Ambulatory-56.

- scratch SBP/DBP MAE: {scratch.sbp_mae:.3f}/{scratch.dbp_mae:.3f} mmHg؛
- pretrained SBP/DBP MAE: {transfer.sbp_mae:.3f}/{transfer.dbp_mae:.3f} mmHg؛
- delta SBP/DBP: {transfer.sbp_mae - scratch.sbp_mae:+.3f}/{transfer.dbp_mae - scratch.dbp_mae:+.3f} mmHg؛
- delta mean MAE: {transfer.mean_mae - scratch.mean_mae:+.3f} mmHg.

مقدار منفی بهتر است. labelهای cuff این دیتاست با waveform هم‌زمان نیستند؛ نتیجه فقط stress test
انتقال دستگاه/cohort است و نباید agreement بالینی تفسیر شود. مدل 10 ثانیه‌ای عمداً روی PPG-BP
دوثانیه‌ای با padding یا تکرار مصنوعی ارزیابی نشد.
"""
    (OUT / "EXTERNAL_RESULTS_FA.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
