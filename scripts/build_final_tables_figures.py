#!/usr/bin/env python3
"""Build reviewer-requested tables and figures from freshly generated artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save_table(name: str, frame: pd.DataFrame, description: str, index: list[dict]) -> None:
    path = TABLES / name
    frame.to_csv(path, index=False)
    index.append({"file": str(path.relative_to(ROOT)), "rows": len(frame), "description": description})


def patient_macro(patient: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    values = np.asarray([
        np.abs(prediction[patient == subject] - truth[patient == subject]).mean(0)
        for subject in np.unique(patient)
    ])
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def diagnostic_row(dataset: str, method: str, patient: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - truth
    macro_sbp, macro_dbp = patient_macro(patient, truth, prediction)
    row = {"dataset": dataset, "method": method, "patients": np.unique(patient).size, "windows": len(truth),
           "patient_macro_mae_sbp": macro_sbp, "patient_macro_mae_dbp": macro_dbp}
    for i, target in enumerate(("sbp", "dbp")):
        residual = error[:, i]
        slope, intercept = np.polyfit(truth[:, i], prediction[:, i], 1)
        row.update({
            f"window_mae_{target}": float(np.abs(residual).mean()),
            f"bias_{target}": float(residual.mean()),
            f"error_sd_{target}": float(residual.std(ddof=1)),
            f"loa_lower_{target}": float(residual.mean() - 1.96 * residual.std(ddof=1)),
            f"loa_upper_{target}": float(residual.mean() + 1.96 * residual.std(ddof=1)),
            f"pearson_{target}": float(np.corrcoef(truth[:, i], prediction[:, i])[0, 1]),
            f"calibration_slope_{target}": float(slope),
            f"calibration_intercept_{target}": float(intercept),
            f"within_5_{target}": float(np.mean(np.abs(residual) <= 5)),
            f"within_10_{target}": float(np.mean(np.abs(residual) <= 10)),
            f"within_15_{target}": float(np.mean(np.abs(residual) <= 15)),
        })
    return row


def bland_altman(ax, truth: np.ndarray, prediction: np.ndarray, target: int, title: str) -> None:
    mean = (truth[:, target] + prediction[:, target]) / 2
    difference = prediction[:, target] - truth[:, target]
    bias, sd = difference.mean(), difference.std(ddof=1)
    # Dense ICU windows are rasterized and lightly subsampled only for rendering.
    select = np.linspace(0, len(mean) - 1, min(len(mean), 8000), dtype=int)
    ax.scatter(mean[select], difference[select], s=3, alpha=.12, rasterized=True)
    ax.axhline(bias, color="black", lw=1.2, label=f"bias {bias:.1f}")
    ax.axhline(bias + 1.96 * sd, color="#b22222", ls="--", lw=1)
    ax.axhline(bias - 1.96 * sd, color="#b22222", ls="--", lw=1)
    ax.set(title=title, xlabel="Mean of reference and estimate (mmHg)", ylabel="Estimate - reference (mmHg)")
    ax.legend(frameon=False, fontsize=8)


def save_figure(fig, name: str, description: str, index: list[dict]) -> None:
    png = FIGURES / f"{name}.png"; pdf = FIGURES / f"{name}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    index.append({"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT)), "description": description})


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    table_index: list[dict] = []; figure_index: list[dict] = []

    split = read_json("data/mimic_bp/splits/official_patient_split.json")
    with np.load(ROOT / "outputs/neural/cache/metadata.npz", allow_pickle=False) as z:
        metadata = {key: z[key].copy() for key in z.files}
    cohort_rows = []
    for part in ("train", "validation", "test"):
        mask = metadata["split"].astype(str) == part
        cohort_rows.append({"cohort": "MIMIC-BP", "partition": part, "patients": np.unique(metadata["patient_id"][mask]).size,
                            "windows_after_duplicate_policy": int(mask.sum()), "reference": "invasive ABP-derived SBP/DBP"})
    vital_manifest = read_json("data/external/vitaldb/prepared/MANIFEST.json")
    cohort_rows.append({"cohort": "VitalDB deterministic synchronized subset", "partition": "locked external",
                        "patients": vital_manifest["accepted_cases"], "windows_after_duplicate_policy": vital_manifest["accepted_windows"],
                        "reference": "timestamp-aligned numeric ART_SBP/ART_DBP medians"})
    save_table("TABLE_1_COHORT_AND_SPLIT.csv", pd.DataFrame(cohort_rows), "Official subject split and external cohort accounting.", table_index)

    confirm = pd.read_csv(ROOT / "phase2/outputs/innovation200_confirmation/OOF_SUMMARY.csv")
    fold_mean = read_json("phase2/outputs/innovation200_confirmation/FOLD_MEAN_BASELINE.json")["metrics"]
    keep = confirm[confirm.candidate_id.isin(["PRIOR_WINNER_FIXED", "ENSEMBLE_BASELINE_ACCURACY_EQUAL"])].copy()
    mean_row = {"candidate_id": "OUTER_TRAINING_MEAN_REPEATED", **fold_mean}
    repeated = pd.concat([pd.DataFrame([mean_row]), keep], ignore_index=True, sort=False)
    save_table("TABLE_2_REPEATED_OOF_DESCRIPTIVE.csv", repeated, "Repeated grouped OOF; descriptive because method selection was not fully nested.", table_index)

    nested = read_json("phase2/outputs/nested200/DECISION.json")
    nested_predictions = np.load(ROOT / "phase2/outputs/nested200/OOF_PREDICTIONS.npz", allow_pickle=False)
    nested_rows = [{"model": name, **values} for name, values in nested["summary"].items()]
    save_table("TABLE_3_NESTED_SELECTION_RESULTS.csv", pd.DataFrame(nested_rows), "Primary five-fold patient-nested estimate of the locked selection procedure.", table_index)
    save_table("TABLE_3B_NESTED_SELECTIONS.csv", pd.DataFrame(nested["selection"]), "Configuration selected independently inside each outer fold.", table_index)

    event = nested["event_safety"]
    event_rows = []
    for endpoint in ("high_bp", "map_below_65"):
        direct = event[endpoint]["direct"]
        event_rows.append({"endpoint": endpoint, "operating_point": "direct numeric output", **direct,
                           "auprc": event[endpoint]["auprc"], "auroc": event[endpoint]["auroc"]})
        event_rows.append({"endpoint": endpoint, "operating_point": "patient-cross-fitted threshold",
                           **event[endpoint]["crossfit_90pct_specificity"], "auprc": event[endpoint]["auprc"], "auroc": event[endpoint]["auroc"]})
    save_table("TABLE_4_EVENT_SAFETY.csv", pd.DataFrame(event_rows), "Operational high-BP and derived-low-MAP event endpoints.", table_index)

    external = pd.read_csv(ROOT / "phase2/outputs/innovation200_vitaldb_external/SUMMARY.csv")
    cuff = pd.read_csv(ROOT / "phase2/outputs/external_validation/calibration/crossfit_calibration_metrics.csv")
    external["dataset"] = "VitalDB"; cuff["dataset"] = cuff["dataset_model"]
    save_table("TABLE_5_EXTERNAL_TRANSFER.csv", pd.concat([external, cuff], ignore_index=True, sort=False),
               "Zero-shot results and clearly separated supervised target-domain adaptation.", table_index)

    strata_rows = []
    for column, (target, edges) in enumerate((("SBP", [-np.inf, 90, 120, 140, 160, np.inf]), ("DBP", [-np.inf, 60, 80, 90, 100, np.inf]))):
        truth = nested_predictions["y_true"][:, column]; prediction = nested_predictions["selected"][:, column]
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (truth >= lower) & (truth < upper)
            error = prediction[mask] - truth[mask]
            strata_rows.append({"target": target, "lower": lower, "upper": upper, "n": int(mask.sum()),
                                "mae": float(np.abs(error).mean()), "bias": float(error.mean()),
                                "true_mean": float(truth[mask].mean()), "predicted_mean": float(prediction[mask].mean())})
    strata = pd.DataFrame(strata_rows)
    save_table("TABLE_6_PRESSURE_RANGE_AUDIT.csv", strata, "Range-specific error and bias; not a causal ablation.", table_index)

    component = pd.read_csv(ROOT / "phase2/outputs/innovation200_search/COMPONENT_EFFECTS.csv")
    registry = read_json("phase2/protocol/INNOVATION200_SEARCH_LOCK.json")
    screening = pd.read_csv(ROOT / "phase2/outputs/innovation200_search/screening_results.csv")
    configs = pd.DataFrame(registry["configs"])
    all_methods = configs.merge(screening, on="run_id", how="left", suffixes=("_config", "_result"))
    save_table("SUPPLEMENT_ALL_200_CONFIGS_AND_RESULTS.csv", all_methods,
               "Complete literature-informed, systematically constructed configuration registry and screening results.", table_index)
    save_table("SUPPLEMENT_COMPONENT_ASSOCIATIONS.csv", component,
               "Descriptive component associations; not causal one-factor ablations.", table_index)
    inner = pd.read_csv(ROOT / "phase2/outputs/nested200/INNER_SCREEN_RESULTS.csv")
    save_table("SUPPLEMENT_NESTED_1005_INNER_FITS.csv", inner, "All 201 candidates in each of five inner selections.", table_index)
    folds = pd.read_csv(ROOT / "phase2/outputs/innovation200_confirmation/fold_results.csv")
    save_table("SUPPLEMENT_REPEATED_75_FITS.csv", folds, "All repeated grouped-confirmation fits.", table_index)

    nested_npz = nested_predictions
    vital_npz = np.load(ROOT / "phase2/outputs/innovation200_vitaldb_external/PREDICTIONS.npz", allow_pickle=False)
    ambulatory_task = np.load(ROOT / "phase2/outputs/external_validation/calibration/crossfit_winner10s_ambulatory56.npz", allow_pickle=False)
    ambulatory_foundation = np.load(ROOT / "phase2/outputs/external_validation/calibration/crossfit_papagei10s_ambulatory56.npz", allow_pickle=False)
    diagnostic = [
        diagnostic_row("MIMIC-BP nested OOF", "nested selected procedure", nested_npz["patient_id"], nested_npz["y_true"], nested_npz["selected"]),
        diagnostic_row("MIMIC-BP nested OOF", "historical fixed prior", nested_npz["patient_id"], nested_npz["y_true"], nested_npz["historical_prior"]),
        diagnostic_row("VitalDB", "zero shot", vital_npz["patient_id"], vital_npz["y_true"], vital_npz["zero_shot"]),
        diagnostic_row("VitalDB", "supervised affine cross-fit", vital_npz["patient_id"], vital_npz["y_true"], vital_npz["affine_crossfit"]),
        diagnostic_row("Ambulatory-56", "task model zero shot", ambulatory_task["patient_id"], ambulatory_task["y_true"], ambulatory_task["zero_shot"]),
        diagnostic_row("Ambulatory-56", "task model supervised affine cross-fit", ambulatory_task["patient_id"], ambulatory_task["y_true"], ambulatory_task["affine_ridge"]),
        diagnostic_row("Ambulatory-56", "PaPaGei zero shot", ambulatory_foundation["patient_id"], ambulatory_foundation["y_true"], ambulatory_foundation["zero_shot"]),
        diagnostic_row("Ambulatory-56", "PaPaGei supervised affine cross-fit", ambulatory_foundation["patient_id"], ambulatory_foundation["y_true"], ambulatory_foundation["affine_ridge"]),
    ]
    save_table("TABLE_7_BIAS_BLAND_ALTMAN_CALIBRATION.csv", pd.DataFrame(diagnostic),
               "Bias, SD, 95% limits of agreement, correlation, and calibration slope/intercept.", table_index)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    bland_altman(axes[0, 0], nested_npz["y_true"], nested_npz["selected"], 0, "MIMIC-BP nested: SBP")
    bland_altman(axes[0, 1], nested_npz["y_true"], nested_npz["selected"], 1, "MIMIC-BP nested: DBP")
    bland_altman(axes[1, 0], vital_npz["y_true"], vital_npz["zero_shot"], 0, "VitalDB zero-shot: SBP")
    bland_altman(axes[1, 1], vital_npz["y_true"], vital_npz["zero_shot"], 1, "VitalDB zero-shot: DBP")
    save_figure(fig, "FIGURE_1_BLAND_ALTMAN", "Bland-Altman plots requested by the reviewer.", figure_index)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for col, target in enumerate(("SBP", "DBP")):
        for row, (prediction, label) in enumerate(((vital_npz["zero_shot"], "Zero-shot"), (vital_npz["affine_crossfit"], "Supervised affine cross-fit"))):
            ax = axes[row, col]; x = vital_npz["y_true"][:, col]; y = prediction[:, col]
            ax.scatter(x, y, s=7, alpha=.18, rasterized=True)
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            slope, intercept = np.polyfit(x, y, 1)
            ax.set(xlabel=f"Reference {target} (mmHg)", ylabel=f"Estimated {target} (mmHg)", title=f"{label}: slope={slope:.2f}")
    save_figure(fig, "FIGURE_2_VITALDB_CALIBRATION", "External target-prediction scatter before and after supervised adaptation.", figure_index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, target in zip(axes, ("SBP", "DBP")):
        values = strata[strata.target == target].copy()
        labels = [f"{lo:g}–{hi:g}" for lo, hi in zip(values.lower, values.upper)]
        ax.bar(labels, values.mae, color="#4472c4"); ax.axhline(values.mae.mean(), color="black", ls=":")
        ax.set(title=f"{target} range-specific error", xlabel="Reference range (mmHg)", ylabel="MAE (mmHg)")
        ax.tick_params(axis="x", rotation=35)
    save_figure(fig, "FIGURE_3_RANGE_SPECIFIC_ERROR", "Aggregate MAE masks severe errors at pressure extremes.", figure_index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(screening.mean_mae, bins=28, color="#70ad47", alpha=.85)
    axes[0].set(title="Locked 200-method screening", xlabel="Validation mean MAE (mmHg)", ylabel="Methods")
    selections = pd.DataFrame(nested["selection"])
    axes[1].bar(selections.outer_fold.astype(str), selections.inner_selection_score, color="#ed7d31")
    axes[1].set(title="Nested inner selections", xlabel="Outer fold", ylabel="Inner normalized score")
    save_figure(fig, "FIGURE_4_SEARCH_AND_NESTED_SELECTION", "Full screening distribution and fold-specific nested selection.", figure_index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    bins = np.linspace(20, 120, 26)
    for ax, source, title in ((axes[0], ambulatory_task, "Task ResNet"), (axes[1], ambulatory_foundation, "PaPaGei")):
        ax.hist(source["y_true"][:, 1], bins=bins, density=True, alpha=.45, label="Reference DBP")
        ax.hist(source["zero_shot"][:, 1], bins=bins, density=True, alpha=.45, label="Zero-shot DBP")
        ax.hist(source["affine_ridge"][:, 1], bins=bins, density=True, histtype="step", lw=2, label="Supervised affine")
        ax.set(title=f"Ambulatory-56: {title}", xlabel="DBP (mmHg)", ylabel="Density")
        ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "FIGURE_5_AMBULATORY_DBP_SHIFT", "Ambulatory DBP distributions expose severe zero-shot output-scale mismatch.", figure_index)

    table_manifest = {"schema_version": "1.0", "duplicate_policy": split["counts"], "tables": table_index}
    (TABLES / "TABLE_INDEX.json").write_text(json.dumps(table_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (FIGURES / "FIGURE_INDEX.json").write_text(json.dumps({"schema_version": "1.0", "figures": figure_index}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": len(table_index), "figures": len(figure_index)}, indent=2))


if __name__ == "__main__":
    main()
