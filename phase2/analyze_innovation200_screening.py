from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "phase2"
SEARCH = P2 / "outputs/innovation200_search"
REPORT = P2 / "reports/INNOVATION200_SCREENING_REPORT_FA.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(row) -> str:
    return (
        f"| `{row.run_id}` | {row.stage} | {row.innovation_label} | "
        f"{row.mae_sbp:.3f} | {row.mae_dbp:.3f} | {row.mean_mae:.3f} | "
        f"{row.delta_mean_vs_reference:+.3f} | {row.high_bp_sensitivity:.3f} | {int(row.parameters):,} |"
    )


def stage_markdown(frame: pd.DataFrame) -> list[str]:
    header = "| stage | runs | mean MAE | best MAE | best score | beat locked mean | GPU min |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, separator]
    for row in frame.itertuples():
        rows.append(
            f"| {row.stage} | {int(row.runs)} | {row.mean_mean_mae:.4f} | "
            f"{row.best_mean_mae:.4f} | {row.best_selection_score:.4f} | "
            f"{int(row.methods_beating_locked_reference_mean)} | {row.runtime_minutes:.2f} |"
        )
    return rows


def main() -> None:
    results = pd.read_csv(SEARCH / "screening_results.csv")
    if len(results) != 200 or results.status.value_counts().to_dict() != {"ok": 200}:
        raise RuntimeError("The 200-run screen is incomplete")
    reference_path = SEARCH / "reference_prior_winner" / "REFERENCE_RESULT.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    confirmation_lock_path = P2 / "protocol/INNOVATION200_CONFIRMATION_LOCK.json"
    confirmation_lock = (
        json.loads(confirmation_lock_path.read_text(encoding="utf-8"))
        if confirmation_lock_path.exists() else None
    )
    if confirmation_lock is not None:
        reference_mean = float(confirmation_lock["reference_mean_mae"])
        reference_integrity = sha256(reference_path) == confirmation_lock["reference_result_sha256"]
        reference_source = "immutable confirmation lock"
    else:
        reference_mean = float(reference["mean_mae"])
        reference_integrity = True
        reference_source = "current reference artifact"
    protocol = json.loads((P2 / "protocol/INNOVATION200_SEARCH_LOCK.json").read_text(encoding="utf-8"))
    config = pd.DataFrame(protocol["configs"])
    config_columns = [column for column in config.columns if column not in results.columns or column == "run_id"]
    merged = results.merge(config[config_columns], on="run_id", how="left", validate="one_to_one")
    # The confirmation lock is the authoritative, pre-confirmation record.  A
    # resumed supervisor must never silently redefine the screening reference.
    merged["delta_mean_vs_reference"] = merged.mean_mae - reference_mean
    merged["beats_locked_reference_mean"] = merged.delta_mean_vs_reference < 0
    merged = merged.sort_values(["selection_score", "clinical_selection_score", "parameters"], kind="stable")
    merged.to_csv(SEARCH / "METHOD_BY_METHOD.csv", index=False)

    factors = [
        "stage", "backbone", "view", "duration_seconds", "engineered", "head", "loss",
        "augmentation", "balance", "model_width", "model_depth", "model_kernel",
        "channel_attention", "pooling", "optimizer", "scheduler",
    ]
    component_rows = []
    for factor in factors:
        if factor not in merged.columns:
            continue
        for value, group in merged.groupby(factor, dropna=False):
            component_rows.append({
                "factor": factor, "value": value, "runs": len(group),
                "mean_sbp_mae": group.mae_sbp.mean(), "mean_dbp_mae": group.mae_dbp.mean(),
                "mean_of_mean_mae": group.mean_mae.mean(), "best_mean_mae": group.mean_mae.min(),
                "mean_clinical_score": group.clinical_selection_score.mean(),
                "mean_high_bp_sensitivity": group.high_bp_sensitivity.mean(),
                "mean_runtime_seconds": group.runtime_seconds.mean(),
            })
    components = pd.DataFrame(component_rows).sort_values(["factor", "mean_of_mean_mae", "runs"])
    components.to_csv(SEARCH / "COMPONENT_EFFECTS.csv", index=False)
    stage = merged.groupby("stage", as_index=False).agg(
        runs=("run_id", "size"), mean_mean_mae=("mean_mae", "mean"),
        best_mean_mae=("mean_mae", "min"), best_selection_score=("selection_score", "min"),
        methods_beating_locked_reference_mean=("beats_locked_reference_mean", "sum"),
        runtime_minutes=("runtime_seconds", lambda x: x.sum() / 60),
    ).sort_values("best_selection_score")
    stage.to_csv(SEARCH / "STAGE_SUMMARY.csv", index=False)

    top = merged.head(20)
    bottom = merged.sort_values("selection_score", ascending=False).head(10)
    lines = [
        "# گزارش اجرای ۲۰۰ روش جدید برای تخمین فشارخون از PPG",
        "",
        "## کنترل کامل‌بودن",
        "",
        f"- روش‌های قفل‌شده: **{len(protocol['configs'])}**",
        f"- اجرای موفق: **{len(merged)}**؛ اجرای شکست‌خورده: **0**",
        f"- میانگین MAE مرجع هم‌seedِ قفل‌شده پیش از confirmation: **`{reference_mean:.3f}`** ({reference_source}).",
        f"- روش‌هایی که mean MAE مرجع قفل‌شده را در screening بهتر کردند: **{int(merged.beats_locked_reference_mean.sum())}**.",
        f"- تطابق فایل replay فعلی با hash مرجع قفل‌شده: **{'PASS' if reference_integrity else 'FAIL؛ replay بعدی مرجع قفل‌شده را بازتولید نکرد و برای رتبه‌بندی استفاده نشد'}**.",
        f"- زمان تجمعی GPU برای ۲۰۰ روش: **{merged.runtime_seconds.sum()/60:.1f} دقیقه**.",
        "",
        "این نتایج فقط screening بیمار-جدا هستند. انتخاب نهایی تنها پس از grouped CV با seedهای تازه و آزمون paired بیمارمحور مجاز است.",
        "",
        "## ۲۰ روش اول",
        "",
        "| run | بلوک | روش | SBP | DBP | mean | Δmean | sens-high | params |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        *[fmt(row) for row in top.itertuples()],
        "",
        "## خلاصهٔ چهار بلوک",
        "",
        *stage_markdown(stage),
        "",
        "## ده روش ضعیف‌تر",
        "",
        "| run | بلوک | روش | SBP | DBP | mean | Δmean | sens-high | params |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        *[fmt(row) for row in bottom.itertuples()],
        "",
        "## تفسیر مجاز",
        "",
        "- اختلاف screening به‌تنهایی نوآوری تأییدشده نیست؛ ۲۰۰ مقایسه احتمال انتخاب تصادفی را بالا می‌برد.",
        "- winner باید در ۵ fold × ۳ seed تازه، در خطای بیمارمحور و فاصلهٔ اطمینان paired از مدل قبلی عبور کند.",
        "- حساسیت فشار بالا و انتقال VitalDB مستقل از MAE داخلی گزارش می‌شوند.",
        "- جدول `METHOD_BY_METHOD.csv` هر ۲۰۰ روش و جدول `COMPONENT_EFFECTS.csv` اثر هر جزء را نگه می‌دارند.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
    print(top[["run_id", "innovation_label", "mae_sbp", "mae_dbp", "mean_mae", "delta_mean_vs_reference"]].to_string(index=False))


if __name__ == "__main__":
    main()
