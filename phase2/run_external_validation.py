from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.engine import load_arrays, run_experiment
from phase2.models import BPModel
from phase2.run_confirmation_cv import SEEDS, subject_strata


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2/outputs/external_validation"
PREPARED = ROOT / "data/external/prepared"


class ExternalDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.from_numpy(np.asarray(self.x[index], dtype=np.float32)), torch.empty(0)


def metric_set(y, pred):
    error = pred - y
    absolute = np.abs(error)
    result = {}
    for i, target in enumerate(("sbp", "dbp")):
        result[f"mae_{target}"] = float(absolute[:, i].mean())
        result[f"rmse_{target}"] = float(np.sqrt((error[:, i] ** 2).mean()))
        result[f"bias_{target}"] = float(error[:, i].mean())
        result[f"error_sd_{target}"] = float(error[:, i].std(ddof=1))
        result[f"pearson_{target}"] = float(np.corrcoef(y[:, i], pred[:, i])[0, 1])
        for limit in (5, 10, 15):
            result[f"pct_{target}_within_{limit}"] = float(100 * (absolute[:, i] <= limit).mean())
    return result


def patient_average(patient, y, pred):
    unique = np.unique(patient)
    y_subject = np.asarray([y[patient == p].mean(0) for p in unique])
    pred_subject = np.asarray([pred[patient == p].mean(0) for p in unique])
    return unique, y_subject, pred_subject


@torch.inference_mode()
def infer_checkpoint(checkpoint_path: Path, x: np.ndarray):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = BPModel(config, 0)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(ExternalDataset(x), batch_size=256, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    predictions = []
    for signal, engineered in loader:
        output = model(signal.to(device, non_blocking=True), engineered.to(device, non_blocking=True))
        normalized = model.point_prediction(output).float().cpu().numpy()
        predictions.append(normalized * checkpoint["target_std"] + checkpoint["target_mean"])
    return np.concatenate(predictions)


def configs():
    winner = json.loads((ROOT / "phase2/outputs/confirmation_cv/winner.json").read_text(encoding="utf-8"))["config"]
    b_registry = json.loads((ROOT / "phase2/outputs/stage_b_resolved_registry.json").read_text(encoding="utf-8"))
    screening = pd.read_csv(ROOT / "phase2/outputs/search_results.csv").set_index("run_id")
    compatible = [c for c in b_registry if float(c["duration_seconds"]) == 2.0 and c.get("engineered", "none") == "none"]
    short = min(compatible, key=lambda c: (float(screening.loc[c["run_id"], "selection_score"]), c["run_id"]))
    winner = winner.copy()
    winner.update({"epochs": 5, "fixed_epochs": True})
    short = short.copy()
    short.update({"epochs": 5, "fixed_epochs": True})
    a_registry = json.loads((ROOT / "phase2/protocol/PROTOCOL_LOCK.json").read_text(encoding="utf-8"))["stage_A"]["configs"]
    foundation = next(c for c in a_registry if c["backbone"] == "papagei" and c.get("engineered") == "none" and c.get("foundation_mode") == "frozen").copy()
    foundation.update({"epochs": 8, "fixed_epochs": True})
    return {"winner_10s": winner, "compatible_2s": short, "papagei_frozen_10s": foundation}


def train_fold_ensemble(tag, base, arrays):
    subjects, strata = subject_strata(arrays["patient_id"], arrays["y"])
    checkpoints = []
    for seed in SEEDS:
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (train_subject_i, test_subject_i) in enumerate(splitter.split(subjects, strata), 1):
            run_id = f"{tag}_seed{seed}_fold{fold}"
            run_dir = OUT / "models" / run_id
            checkpoint = run_dir / "best_model.pt"
            if not checkpoint.exists():
                train_subjects, test_subjects = subjects[train_subject_i], subjects[test_subject_i]
                train_idx = np.flatnonzero(np.isin(arrays["patient_id"], train_subjects))
                test_idx = np.flatnonzero(np.isin(arrays["patient_id"], test_subjects))
                config = base.copy()
                config.update({"run_id": run_id, "stage": "EXTERNAL_MODEL"})
                run_experiment(config, arrays, train_idx, test_idx, run_dir, seed=seed, save_model=True)
            checkpoints.append(checkpoint)
    return checkpoints


def evaluate(tag, checkpoints, dataset_name, dataset_path):
    with np.load(dataset_path, allow_pickle=False) as data:
        x, y, patient = data["x"].copy(), data["y"].copy(), data["patient_id"].copy()
    member_predictions = np.asarray([infer_checkpoint(path, x) for path in checkpoints])
    prediction = member_predictions.mean(0)
    subject, y_subject, p_subject = patient_average(patient, y, prediction)
    result = {
        "model": tag,
        "dataset": dataset_name,
        "ensemble_members": len(checkpoints),
        "records": len(y),
        "subjects": len(subject),
        "evaluation_unit": "subject",
        **metric_set(y_subject, p_subject),
    }
    np.savez_compressed(OUT / f"predictions_{tag}_{dataset_name}.npz", patient_id=subject, y_true=y_subject, y_pred=p_subject, member_record_predictions=member_predictions)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays()
    model_configs = configs()
    ensembles = {tag: train_fold_ensemble(tag, config, arrays) for tag, config in model_configs.items()}
    rows = []
    rows.append(evaluate("winner_10s", ensembles["winner_10s"], "ambulatory56", PREPARED / "ppg_ambulatory_56.npz"))
    rows.append(evaluate("papagei_frozen_10s", ensembles["papagei_frozen_10s"], "ambulatory56", PREPARED / "ppg_ambulatory_56.npz"))
    rows.append(evaluate("compatible_2s", ensembles["compatible_2s"], "ambulatory56", PREPARED / "ppg_ambulatory_56.npz"))
    rows.append(evaluate("compatible_2s", ensembles["compatible_2s"], "ppgbp219", PREPARED / "ppg_bp_219.npz"))
    pd.DataFrame(rows).to_csv(OUT / "external_metrics.csv", index=False)
    (OUT / "EXTERNAL_RESULTS.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
