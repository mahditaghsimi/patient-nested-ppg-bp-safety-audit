#!/usr/bin/env python3
"""Print and save article-ready headline results and the output inventory."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    nested = read("phase2/outputs/nested200/DECISION.json")
    repeated = read("phase2/outputs/innovation200_confirmation/DECISION.json")
    vital = read("phase2/outputs/innovation200_vitaldb_external/DECISION.json")
    n = nested["summary"]["nested_selected_procedure"]
    p = nested["summary"]["historical_fixed_prior"]
    m = nested["summary"]["outer_fold_training_mean"]
    z = vital["zero_shot"]
    s = vital["source_mean"]
    a = vital["supervised_affine_context"]
    figures = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "results/figures").glob("*.png"))
    tables = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "results/tables").glob("*.csv"))
    lines = [
        "# FINAL article results", "", "## Primary patient-nested evaluation", "",
        f"- Selected procedure MAE (SBP/DBP): {n['mae_sbp']:.3f}/{n['mae_dbp']:.3f} mmHg",
        f"- Historical fixed prior MAE: {p['mae_sbp']:.3f}/{p['mae_dbp']:.3f} mmHg",
        f"- Outer-training mean MAE: {m['mae_sbp']:.3f}/{m['mae_dbp']:.3f} mmHg",
        f"- Direct high-BP sensitivity: {100*n['high_bp_sensitivity']:.2f}%",
        f"- Selected vs historical prior p-value: {nested['paired_patient_tests']['selected_vs_historical_prior']['wilcoxon_p']:.6g}",
        "", "## Secondary repeated grouped evaluation", "",
        f"- Decision: {repeated['status']}", f"- Winner: {repeated['winner_id']}",
        f"- MAE: {repeated['winner_oof']['mae_sbp']:.3f}/{repeated['winner_oof']['mae_dbp']:.3f} mmHg",
        "", "## VitalDB external stress test", "",
        f"- External model: {vital['candidate_id']} (not the fold-varying nested procedure)",
        f"- Zero-shot MAE: {z['mae_sbp']:.3f}/{z['mae_dbp']:.3f} mmHg",
        f"- Official source-training mean MAE: {s['mae_sbp']:.3f}/{s['mae_dbp']:.3f} mmHg",
        f"- Supervised cross-fitted affine MAE: {a['mae_sbp']:.3f}/{a['mae_dbp']:.3f} mmHg",
        "", "## Tables", "", *[f"- `{path}`" for path in tables],
        "", "## Figures", "", *[f"- `{path}`" for path in figures],
        "", "## Claim boundary", "",
        "Retrospective research only. These results do not validate a diagnostic or monitoring device.",
    ]
    text = "\n".join(lines) + "\n"
    destination = ROOT / "artifacts/FINAL_SUMMARY.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

