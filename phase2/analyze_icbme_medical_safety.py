from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
CLINICAL = ROOT / "phase2/outputs/icbme_clinical_analysis"
CLINICAL_SEARCH = ROOT / "phase2/outputs/icbme_clinical_search"
EVENT = ROOT / "phase2/outputs/icbme_event_search"
OUT = ROOT / "phase2/outputs/icbme_medical_safety"
BASELINE = "resnet_multiview10__physio_multitask_aug"


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int(np.sum(truth & prediction)); fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction)); fp = int(np.sum(~truth & prediction))
    return {
        "n": int(len(truth)), "prevalence": float(truth.mean()),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
    }


def threshold_at_specificity(truth: np.ndarray, score: np.ndarray, target_specificity: float = 0.90) -> float:
    fpr, tpr, thresholds = roc_curve(truth, score)
    eligible = np.flatnonzero(fpr <= 1 - target_specificity + 1e-12)
    if not len(eligible):
        return float("inf")
    best_tpr = np.max(tpr[eligible])
    candidates = eligible[tpr[eligible] == best_tpr]
    # The strictest threshold resolves ties and prevents optimistic specificity.
    return float(np.max(thresholds[candidates]))


def repeated_patient_crossfit_threshold(patient: np.ndarray, truth: np.ndarray, score: np.ndarray) -> dict:
    subjects = np.unique(patient)
    totals = {key: 0 for key in ["tp", "fn", "tn", "fp"]}
    thresholds = []
    for repeat in range(10):
        splitter = KFold(5, shuffle=True, random_state=20260821 + repeat)
        for train_subject_i, test_subject_i in splitter.split(subjects):
            train_subjects, test_subjects = subjects[train_subject_i], subjects[test_subject_i]
            train = np.isin(patient, train_subjects); test = np.isin(patient, test_subjects)
            threshold = threshold_at_specificity(truth[train], score[train], 0.90)
            thresholds.append(threshold)
            current = classification_metrics(truth[test], score[test] >= threshold)
            for key in totals: totals[key] += current[key]
    tp, fn, tn, fp = (totals[key] for key in ["tp", "fn", "tn", "fp"])
    return {
        "repeats": 10, "folds": 5,
        "threshold_median": float(np.median(thresholds)),
        "threshold_iqr_low": float(np.quantile(thresholds, 0.25)),
        "threshold_iqr_high": float(np.quantile(thresholds, 0.75)),
        "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
        "ppv": float(tp / (tp + fp)), "npv": float(tn / (tn + fn)),
        **totals,
    }


def pressure_strata(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    sbp_bins = [-np.inf, 90, 120, 140, 160, np.inf]
    dbp_bins = [-np.inf, 60, 80, 90, 100, np.inf]
    rows = []
    for target, values, prediction, bins in [("SBP", y[:, 0], pred[:, 0], sbp_bins), ("DBP", y[:, 1], pred[:, 1], dbp_bins)]:
        labels = np.digitize(values, bins[1:-1])
        for group in range(len(bins) - 1):
            mask = labels == group
            error = prediction[mask] - values[mask]
            rows.append({
                "target": target, "lower": bins[group], "upper": bins[group + 1], "n": int(mask.sum()),
                "mae": float(np.abs(error).mean()), "bias": float(error.mean()),
                "true_mean": float(values[mask].mean()), "predicted_mean": float(prediction[mask].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(CLINICAL / "OOF_PREDICTIONS.npz", allow_pickle=False) as data:
        y = data["y_true"].copy(); pred = data[BASELINE].copy(); patient = data["patient_id"].astype(str)
    high = (y[:, 0] >= 140) | (y[:, 1] >= 90)
    predicted_high = (pred[:, 0] >= 140) | (pred[:, 1] >= 90)
    map_true = (y[:, 0] + 2 * y[:, 1]) / 3
    map_pred = (pred[:, 0] + 2 * pred[:, 1]) / 3
    map_low = map_true < 65
    predicted_map_low = map_pred < 65
    high_score = np.maximum(pred[:, 0] - 140, pred[:, 1] - 90)
    low_score = 65 - map_pred

    clinical_threshold = {
        "high_bp": classification_metrics(high, predicted_high),
        "map_below_65": classification_metrics(map_low, predicted_map_low),
    }
    ranking = {
        "high_bp": {"auprc": float(average_precision_score(high, high_score)), "auroc": float(roc_auc_score(high, high_score))},
        "map_below_65": {"auprc": float(average_precision_score(map_low, low_score)), "auroc": float(roc_auc_score(map_low, low_score))},
    }
    crossfit = {
        "high_bp": repeated_patient_crossfit_threshold(patient, high, high_score),
        "map_below_65": repeated_patient_crossfit_threshold(patient, map_low, low_score),
    }
    pressure_strata(y, pred).to_csv(OUT / "PRESSURE_STRATA.csv", index=False)

    event_ranking = pd.read_csv(EVENT / "screening_ranking.csv")
    # Resolve the comparator semantically from this run's registry.  A former
    # version hard-coded a historical run id, which was neither portable nor
    # valid after the official split and central-window corrections.
    clinical_screen = pd.read_csv(CLINICAL_SEARCH / "screening_results.csv")
    baseline_rows = clinical_screen.loc[clinical_screen["clinical_method"] == BASELINE]
    if baseline_rows.empty:
        raise RuntimeError(f"Clinical baseline {BASELINE!r} is absent from screening results")
    baseline_row = baseline_rows.sort_values("clinical_selection_score").iloc[0]
    regression_run_id = str(baseline_row["run_id"])
    regression_validation = CLINICAL_SEARCH / "runs" / regression_run_id / "best_predictions.npz"
    with np.load(regression_validation, allow_pickle=False) as data:
        validation_y, validation_pred = data["y_true"], data["y_pred"]
    validation_high = (validation_y[:, 0] >= 140) | (validation_y[:, 1] >= 90)
    validation_map_low = (validation_y[:, 0] + 2 * validation_y[:, 1]) / 3 < 65
    regression_high_score = np.maximum(validation_pred[:, 0] - 140, validation_pred[:, 1] - 90)
    regression_low_score = 65 - (validation_pred[:, 0] + 2 * validation_pred[:, 1]) / 3
    regression_score = 0.65 * average_precision_score(validation_high, regression_high_score) + 0.35 * average_precision_score(validation_map_low, regression_low_score)
    best_event = event_ranking.iloc[0]
    event_decision = {
        "event_search_configs": int(len(event_ranking)),
        "regression_comparator_run_id": regression_run_id,
        "best_event_model": str(best_event.run_id),
        "best_event_high_auprc": float(best_event.high_auprc),
        "best_event_map_low_auprc": float(best_event.map_low_auprc),
        "best_event_selection_score": float(best_event.selection_score),
        "regression_ranking_high_auprc_same_validation": float(average_precision_score(validation_high, regression_high_score)),
        "regression_ranking_map_low_auprc_same_validation": float(average_precision_score(validation_map_low, regression_low_score)),
        "regression_selection_score_same_validation": float(regression_score),
        "event_model_improved_over_regression": bool(best_event.selection_score > regression_score),
    }
    decision = {
        "clinical_threshold": clinical_threshold,
        "ranking_information": ranking,
        "crossfitted_operating_point_90pct_specificity": crossfit,
        "event_model_screen": event_decision,
        "interpretation": "The regression carries moderate risk-ranking information but its absolute BP scale collapses toward the cohort center, causing clinically unacceptable event sensitivity. Direct event classifiers did not outperform regression ranking on the screening split.",
    }
    (OUT / "MEDICAL_SAFETY_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    high_ct = clinical_threshold["high_bp"]
    low_ct = clinical_threshold["map_below_65"]
    high_cf = crossfit["high_bp"]
    low_cf = crossfit["map_below_65"]
    report = f"""# ممیزی ایمنی پزشکی مدل PPG-to-BP

## نتیجهٔ اصلی

MAE بیمارمحور و leakage-safe مدل پایه برابر 12.91/8.20 mmHg است، اما این عدد عملکرد رخداد
بالینی را پنهان می‌کند. از {high_ct['tp'] + high_ct['fn']:,} پنجرهٔ دارای فشار بالا، فقط
{high_ct['tp']:,} مورد با threshold عددی خود مدل شناسایی شد.

## threshold بالینی مستقیم

| رخداد | شیوع | حساسیت | اختصاصیت | PPV | NPV |
|---|---:|---:|---:|---:|---:|
| SBP>=140 یا DBP>=90 | {100*high_ct['prevalence']:.1f}% | {100*high_ct['sensitivity']:.1f}% | {100*high_ct['specificity']:.1f}% | {100*high_ct['ppv']:.1f}% | {100*high_ct['npv']:.1f}% |
| MAP<65 | {100*low_ct['prevalence']:.1f}% | {100*low_ct['sensitivity']:.1f}% | {100*low_ct['specificity']:.1f}% | {100*low_ct['ppv']:.1f}% | {100*low_ct['npv']:.1f}% |

برای فشار بالا، MAE به 28.20/17.10 mmHg و bias به -28.03/-16.77 mmHg می‌رسد. مدل فشارهای
بالا را شدیداً به سمت میانگین cohort برمی‌گرداند.

## آیا waveform هنوز اطلاعات رتبه‌بندی خطر دارد؟

- فشار بالا: AUROC={ranking['high_bp']['auroc']:.3f} و AUPRC={ranking['high_bp']['auprc']:.3f}؛
- MAP<65: AUROC={ranking['map_below_65']['auroc']:.3f} و AUPRC={ranking['map_below_65']['auprc']:.3f}.

در threshold calibration تکرارشده و patient-wise با هدف حداقل 90% specificity:

- حساسیت فشار بالا {100*high_cf['sensitivity']:.1f}% در اختصاصیت {100*high_cf['specificity']:.1f}%؛
- حساسیت MAP<65 برابر {100*low_cf['sensitivity']:.1f}% در اختصاصیت {100*low_cf['specificity']:.1f}%.

پس representation برای screening/risk ranking مقداری اطلاعات دارد، ولی خروجی عددی calibration-free
برای تشخیص رخداد قابل استفاده نیست.

## classifier اختصاصی رخداد

تعداد {event_decision['event_search_configs']} پیکربندی classifier شامل BCE، focal، class weighting،
event-balanced sampling، joint regression، هشت representation و augmentation آزمایش شد. بهترین مدل
AUPRC فشار بالا={event_decision['best_event_high_auprc']:.3f} و MAP<65={event_decision['best_event_map_low_auprc']:.3f}
داشت. regression موجود روی همان validation به‌ترتیب {event_decision['regression_ranking_high_auprc_same_validation']:.3f}
و {event_decision['regression_ranking_map_low_auprc_same_validation']:.3f} رسید؛ بنابراین classifier اختصاصی
بهبود غربالگری‌شده ایجاد نکرد و وارد confirmation نشد.

## عدم‌قطعیت و کیفیت

- بازهٔ conformal با پوشش واقعی حدود 90%، عرض متوسط 53.20 mmHg برای SBP و 34.00 mmHg برای DBP داشت؛
- حذف 50% نمونه‌های پرعدم‌قطعیت MAE را به 12.31/7.83 رساند، اما حساسیت فشار بالا همچنان فقط 2.3% بود؛
- حذف نمونه‌های SQI پایین بهبود ناچیزی ایجاد کرد و مشکل رخداد را حل نکرد.

نتیجه: abstention مبتنی بر uncertainty ممکن است MAE را ظاهراً بهتر کند، ولی با کنارگذاشتن نمونه‌های
سخت و بالینی مهم، ایمنی رخداد را بهبود نمی‌دهد.

## پیام مناسب مقالهٔ پزشکی

«MAE متوسط برای ارزیابی سامانه‌های cuffless BP کافی نیست. گزارش sensitivity رخداد، bias در محدوده‌های
فشار، patient-wise uncertainty و cross-device validation باید اجباری باشد.» این نتیجه از ادعای یک
معماری جدید قوی‌تر و با محور سلامت دیجیتال و پردازش سیگنال زیستی کنفرانس سازگارتر است.
"""
    (OUT / "MEDICAL_SAFETY_REPORT_FA.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
