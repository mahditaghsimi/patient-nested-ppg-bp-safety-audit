from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "phase2"
SEARCH = P2 / "outputs/innovation200_search"
PROTOCOL = P2 / "protocol/INNOVATION200_SEARCH_LOCK.json"
LOCK = P2 / "protocol/INNOVATION200_CONFIRMATION_LOCK.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_distinct(frame: pd.DataFrame, selected: list[str], predicate=None) -> str:
    candidates = frame if predicate is None else frame[predicate(frame)]
    for row in candidates.itertuples():
        if row.run_id not in selected:
            return row.run_id
    raise RuntimeError("No distinct finalist satisfies the locked selection rule")


def main() -> None:
    if LOCK.exists():
        print(LOCK)
        print("confirmation lock already exists; refusing retrospective reselection")
        return
    results_path = SEARCH / "screening_results.csv"
    ranking_path = SEARCH / "screening_ranking.csv"
    reference_path = SEARCH / "reference_prior_winner" / "REFERENCE_RESULT.json"
    for path in (results_path, ranking_path, reference_path, PROTOCOL):
        if not path.exists():
            raise FileNotFoundError(path)
    results = pd.read_csv(results_path)
    if len(results) != 200 or results.status.value_counts().to_dict() != {"ok": 200}:
        raise RuntimeError(f"Screening incomplete: rows={len(results)}, status={results.status.value_counts().to_dict()}")
    ranking = pd.read_csv(ranking_path).sort_values(
        ["selection_score", "clinical_selection_score", "mean_mae", "parameters"], kind="stable"
    )
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    by_id = {row["run_id"]: row for row in protocol["configs"]}
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    accuracy_gate = float(reference["mean_mae"]) + 0.15
    strict = ranking[ranking.mean_mae <= accuracy_gate].copy()
    if strict.empty:
        raise RuntimeError("No new method passed the reference+0.15 mmHg safety gate")

    selected: list[str] = []
    roles: dict[str, str] = {}
    best_accuracy = pick_distinct(strict, selected)
    selected.append(best_accuracy); roles[best_accuracy] = "best_normalized_accuracy"

    clinical = strict.sort_values(["clinical_selection_score", "selection_score", "parameters"], kind="stable")
    best_clinical = pick_distinct(clinical, selected)
    selected.append(best_clinical); roles[best_clinical] = "best_tail_aware_accuracy"

    event = strict.sort_values(
        ["high_bp_sensitivity", "mean_mae", "clinical_selection_score"],
        ascending=[False, True, True], kind="stable",
    )
    best_event = pick_distinct(event, selected)
    selected.append(best_event); roles[best_event] = "best_direct_high_bp_sensitivity_under_accuracy_gate"

    efficient_pool = strict[strict.parameters <= 500_000].sort_values(
        ["selection_score", "parameters", "clinical_selection_score"], kind="stable"
    )
    if efficient_pool.empty:
        efficient_pool = strict.sort_values(["parameters", "selection_score"], kind="stable")
    best_efficient = pick_distinct(efficient_pool, selected)
    selected.append(best_efficient); roles[best_efficient] = "best_parameter_efficient_under_accuracy_gate"

    baseline = {
        "run_id": "PRIOR_WINNER_FIXED",
        "stage": "REFERENCE",
        "innovation_label": "prior_resnet_multiview10_multitask",
        "selection_role": "fixed_prior_winner",
        "screening_epoch": 5,
        "backbone": "resnet", "view": "multiview", "duration_seconds": 10.0,
        "engineered": "none", "head": "multitask", "loss": "multitask",
        "augmentation": "jitter_scale", "balance": "none", "dropout": 0.15,
        "head_hidden": 192, "epochs": 5, "patience": 2, "batch_size": 192,
        "learning_rate": 1e-3, "weight_decay": 2e-4,
        "optimizer": "adamw", "scheduler": "none",
    }
    finalists = [baseline]
    screening_rows = ranking.set_index("run_id")
    for run_id in selected:
        config = dict(by_id[run_id])
        config["selection_role"] = roles[run_id]
        config["screening_epoch"] = int(screening_rows.loc[run_id, "epoch"])
        config["screening_mae_sbp"] = float(screening_rows.loc[run_id, "mae_sbp"])
        config["screening_mae_dbp"] = float(screening_rows.loc[run_id, "mae_dbp"])
        config["screening_selection_score"] = float(screening_rows.loc[run_id, "selection_score"])
        finalists.append(config)

    payload = {
        "locked_after_complete_screening_before_confirmation": True,
        "screening_protocol_sha256": sha256(PROTOCOL),
        "screening_results_sha256": sha256(results_path),
        "reference_result_sha256": sha256(reference_path),
        "screening_methods_completed": 200,
        "reference_mean_mae": float(reference["mean_mae"]),
        "accuracy_gate_mean_mae": accuracy_gate,
        "selection_rule": [
            "best normalized accuracy",
            "best clinical tail-aware score",
            "best direct high-BP sensitivity under accuracy gate",
            "best <=500k parameter method under accuracy gate",
        ],
        "confirmation": {
            "folds": 5,
            "fresh_seeds": [2718, 57721, 104729],
            "patient_separated": True,
            "fixed_epochs_from_screening": True,
            "outer_fold_epoch_selection": "forbidden",
        },
        "finalists": finalists,
    }
    LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame([{
        "run_id": row["run_id"], "role": row["selection_role"],
        "backbone": row["backbone"], "view": row["view"], "head": row["head"],
        "screening_mae_sbp": row.get("screening_mae_sbp", reference["mae_sbp"]),
        "screening_mae_dbp": row.get("screening_mae_dbp", reference["mae_dbp"]),
    } for row in finalists])
    table.to_csv(SEARCH / "SELECTED_FINALISTS.csv", index=False)
    print(LOCK)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
