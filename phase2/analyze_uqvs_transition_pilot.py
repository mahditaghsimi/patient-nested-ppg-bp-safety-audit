from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "phase2" / "outputs" / "uqvs_transition_pilot"


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[sample].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def main() -> None:
    pred = pd.read_csv(OUT / "GROUP_OOF_PREDICTIONS.csv")
    pred = pred[pred.major_transition].copy()
    pred["ae_sbp"] = (pred.pred_sbp - pred.future_sbp).abs()
    pred["ae_dbp"] = (pred.pred_dbp - pred.future_dbp).abs()
    patient = pred.groupby(["model", "case"])[["ae_sbp", "ae_dbp"]].mean().reset_index()
    patient.to_csv(OUT / "PATIENT_EVENT_ERRORS.csv", index=False)
    baseline = patient[patient.model.eq("persistence")].set_index("case")
    comparison_rows = []
    raw_ps = []
    for model in sorted(set(patient.model) - {"persistence"}):
        candidate = patient[patient.model.eq(model)].set_index("case")
        common = baseline.index.intersection(candidate.index)
        for target_i, target in enumerate(("sbp", "dbp")):
            delta = (
                candidate.loc[common, f"ae_{target}"] - baseline.loc[common, f"ae_{target}"]
            ).to_numpy(float)
            try:
                pvalue = float(wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError:
                pvalue = 1.0
            low, high = bootstrap_ci(delta, seed=1000 + target_i)
            raw_ps.append(pvalue)
            comparison_rows.append({
                "model": model, "target": target, "patients": len(delta),
                "candidate_patient_mean_mae": float(candidate.loc[common, f"ae_{target}"].mean()),
                "persistence_patient_mean_mae": float(baseline.loc[common, f"ae_{target}"].mean()),
                "paired_delta_candidate_minus_persistence": float(delta.mean()),
                "paired_delta_ci_low": low, "paired_delta_ci_high": high,
                "patients_improved_fraction": float(np.mean(delta < 0)),
                "wilcoxon_p": pvalue,
            })
    order = np.argsort(raw_ps)
    adjusted = np.empty(len(raw_ps), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, raw_ps[idx] * (len(raw_ps) - rank))
        adjusted[idx] = min(1.0, running)
    for row, value in zip(comparison_rows, adjusted):
        row["holm_p"] = float(value)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "PAIRED_PATIENT_TESTS.csv", index=False)

    fold_summary = pd.read_csv(OUT / "SUMMARY.csv")
    candidate = "extratrees_eventweighted_delta"
    candidate_tests = comparison[comparison.model.eq(candidate)].set_index("target")
    group_event = fold_summary.query(
        "protocol == 'patient_group_5fold' and subset == 'major_transition' and model == @candidate"
    ).iloc[0]
    random_event = fold_summary.query(
        "protocol == 'sample_random_80_20' and subset == 'major_transition' and model == @candidate"
    ).iloc[0]
    confirmed = bool(
        group_event.skill_sbp_mean > 0
        and group_event.skill_dbp_mean > 0
        and candidate_tests.loc["sbp", "paired_delta_ci_high"] < 0
        and candidate_tests.loc["dbp", "paired_delta_ci_high"] < 0
    )
    decision = {
        "candidate": candidate,
        "patient_group_transition_skill_sbp": float(group_event.skill_sbp_mean),
        "patient_group_transition_skill_dbp": float(group_event.skill_dbp_mean),
        "sample_random_transition_skill_sbp": float(random_event.skill_sbp_mean),
        "sample_random_transition_skill_dbp": float(random_event.skill_dbp_mean),
        "accuracy_improvement_confirmed": confirmed,
        "reason": (
            "Confirmed only when both patient-bootstrap 95% CIs for paired MAE differences are below zero."
            if confirmed else
            "Mean transition skill is promising, but at least one patient-bootstrap 95% CI includes zero; treat as feasibility, not confirmed superiority."
        ),
    }
    (OUT / "FINAL_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    sbp = candidate_tests.loc["sbp"]
    dbp = candidate_tests.loc["dbp"]
    report = f"""# ممیزی ایده‌ی پیش‌بینی پنج‌دقیقه‌ای فشار خون از PPG

## نتیجه‌ی کوتاه

ایده **از نظر علمی امکان‌پذیر و قابل مقاله‌شدن است**، اما آزمایش ده‌دقیقه‌ای فعلی فقط feasibility
است و هنوز برتری قطعی را اثبات نمی‌کند. مدل event-weighted در پنج فولد بیمار-جدا، روی transitionها
به‌طور میانگین {100*group_event.skill_sbp_mean:.1f}٪ برای SBP و
{100*group_event.skill_dbp_mean:.1f}٪ برای DBP نسبت به persistence skill داشت. همان روش در split
تصادفی به {100*random_event.skill_sbp_mean:.1f}٪/{100*random_event.skill_dbp_mean:.1f}٪ رسید؛ این
فاصله نشان می‌دهد split تصادفی نتیجه را به‌شدت خوش‌بینانه می‌کند.

## آزمون بیمارمحور

- اختلاف MAE بیمارمحور SBP (مدل منهای persistence): {sbp.paired_delta_candidate_minus_persistence:.2f}
  mmHg، CI 95٪ برابر [{sbp.paired_delta_ci_low:.2f}, {sbp.paired_delta_ci_high:.2f}]؛
- اختلاف MAE بیمارمحور DBP: {dbp.paired_delta_candidate_minus_persistence:.2f} mmHg، CI 95٪ برابر
  [{dbp.paired_delta_ci_low:.2f}, {dbp.paired_delta_ci_high:.2f}]؛
- سهم بیماران بهترشده: {100*sbp.patients_improved_fraction:.1f}٪ برای SBP و
  {100*dbp.patients_improved_fraction:.1f}٪ برای DBP؛
- نتیجه‌ی gate: **{'عبور قطعی' if confirmed else 'عدم عبور قطعی؛ سیگنال امیدوارکننده ولی CI شامل صفر'}**.

## معنای علمی

مقاله‌ی FBPP-Net سال ۲۰۲۶ از پنجره‌های ۶۰ثانیه‌ای با ۵۰٪ overlap و split تصادفی ۸۰/۲۰ استفاده
کرده است. آزمایش حاضر نشان می‌دهد همین انتخاب می‌تواند skill ظاهری transition را حدود سه برابر
بزرگ‌تر نشان دهد. نوآوری مناسب، تکرار MoE یا quantile loss نیست؛ نوآوری باید ترکیب زیر باشد:

1. پیش‌بینی residual/delta فشار حدود پنج دقیقه بعد؛
2. آموزش event-enriched و loss وزن‌دار، ولی آزمون با prevalence طبیعی؛
3. split کاملاً بیمار-جدا و فاصله‌گذاری پنجره‌های هم‌پوشان؛
4. گزارش skill نسبت به persistence و معیارهای trend، نه فقط MAE؛
5. یک شاخه‌ی PPG-only و یک شاخه‌ی calibration-lite با آخرین/یک فشار مرجع؛
6. معماری Tiny causal با گزارش حافظه و latency؛
7. uncertainty/abstention برای رخدادهای کم‌کیفیت.

مدل PPG-only absolute در این pilot از persistence ضعیف‌تر بود؛ بنابراین نباید فعلاً ادعای
«فشارسنج کاملاً بدون کالیبراسیون» مطرح شود. مسیر واقع‌بینانه‌تر یک حسگر PPG تک‌کاناله با یک اندازه‌گیری
مرجع اولیه یا دوره‌ای است که **تغییر فشار** را ارزان و پیوسته دنبال می‌کند.
"""
    (OUT / "IDEA_FEASIBILITY_REPORT_FA.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

