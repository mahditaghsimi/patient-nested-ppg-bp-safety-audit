from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "phase2"
OUT = P2 / "reports"
FIG = OUT / "figures"
TABLE = OUT / "tables"


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()


def architecture_figure(search):
    a = search[(search.stage == "A") & search.selection_score.notna()].copy()
    a = a.sort_values("selection_score")
    plt.figure(figsize=(9, 6))
    colors = ["#2c7fb8" if name != "papagei" else "#d95f0e" for name in a.backbone]
    plt.barh(a.run_id, a.selection_score, color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Validation selection score (lower is better)")
    plt.title("Stage A: architecture and foundation-model screen")
    plt.axvline(a.selection_score.min(), color="black", linestyle="--", linewidth=1)
    savefig("01_stage_a_architectures.png")


def block_figure(search):
    b = search[(search.stage == "B") & search.selection_score.notna()].copy()
    summary = b.groupby(["duration_seconds", "view"], as_index=False).selection_score.mean()
    plt.figure(figsize=(8, 5))
    for view, group in summary.groupby("view"):
        plt.plot(group.duration_seconds, group.selection_score, marker="o", label=view)
    plt.xlabel("Input duration (s)")
    plt.ylabel("Mean validation selection score")
    plt.title("Stage B: duration and signal-view effects")
    plt.legend(fontsize=8)
    savefig("02_duration_view_ablation.png")


def objective_figure(search):
    c = search[(search.stage == "C") & search.selection_score.notna()].copy()
    c["objective"] = c["head"].astype(str) + "+" + c["loss"].astype(str) + "+" + c["augmentation"].astype(str)
    summary = c.groupby("objective", as_index=False).selection_score.mean().sort_values("selection_score")
    plt.figure(figsize=(9, 5))
    plt.barh(summary.objective, summary.selection_score, color="#41ab5d")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean validation selection score")
    plt.title("Stage C: objective/head/augmentation blocks")
    savefig("03_objective_ablation.png")


def internal_figure(oof):
    labels = ["5s multi-task", "10s multi-view multi-task", "10s multi-view L1", "Equal ensemble"]
    x = np.arange(len(oof))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, oof.mae_sbp, width, label="SBP", color="#3182bd")
    plt.bar(x + width / 2, oof.mae_dbp, width, label="DBP", color="#e6550d")
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylabel("OOF MAE (mmHg)")
    plt.title("Repeated grouped-CV out-of-fold performance")
    plt.legend()
    savefig("04_confirmation_oof.png")


def bland_altman():
    with np.load(P2 / "outputs/analysis/oof_predictions.npz") as data:
        y, pred = data["y_true"], data["simple_ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, target in enumerate(("SBP", "DBP")):
        mean = (y[:, i] + pred[:, i]) / 2
        difference = pred[:, i] - y[:, i]
        bias, sd = difference.mean(), difference.std(ddof=1)
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(mean), size=min(8000, len(mean)), replace=False)
        axes[i].scatter(mean[chosen], difference[chosen], s=3, alpha=0.12)
        axes[i].axhline(bias, color="black")
        axes[i].axhline(bias + 1.96 * sd, color="red", linestyle="--")
        axes[i].axhline(bias - 1.96 * sd, color="red", linestyle="--")
        axes[i].set(title=f"{target}: bias={bias:.2f}, SD={sd:.2f}", xlabel="Mean measured/predicted (mmHg)", ylabel="Prediction error (mmHg)")
    savefig("05_bland_altman_oof.png")


def external_figure(calibration):
    keep = calibration[calibration.method.isin(["zero_shot", "affine_ridge", "mean_baseline"])].copy()
    labels = {
        "winner10s_ambulatory56": "Task ResNet / Amb56",
        "papagei10s_ambulatory56": "PaPaGei / Amb56",
        "short2s_ambulatory56": "2s ResNet / Amb56",
        "short2s_ppgbp219": "2s ResNet / PPG-BP",
    }
    keep["label"] = keep.dataset_model.map(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = ["zero_shot", "affine_ridge", "mean_baseline"]
    colors = ["#cb181d", "#238b45", "#969696"]
    x = np.arange(len(labels))
    for j, target in enumerate(("sbp", "dbp")):
        for i, method in enumerate(methods):
            values = [float(keep[(keep.label == label) & (keep.method == method)][f"mae_{target}"].iloc[0]) for label in labels.values()]
            axes[j].bar(x + (i - 1) * 0.25, values, 0.25, label=method, color=colors[i])
        axes[j].set_xticks(x, labels.values(), rotation=20, ha="right")
        axes[j].set_ylabel(f"{target.upper()} MAE (mmHg)")
        axes[j].set_title(f"External {target.upper()} transfer")
    axes[0].legend(fontsize=8)
    savefig("06_external_transfer.png")


def literature_matrix():
    rows = [
        ["Trustworthy AI review", 2026, "systematic review", "mixed", "Only 34 subject-wise studies; split leakage widespread", "patient-wise evaluation, external validation, distribution-aware metrics"],
        ["BP benchmark", 2023, "11 ML/DL methods; 4 datasets", "subject-wise CV", "Calibration-free PPG often near simple baselines", "common split, classical and deep baselines"],
        ["LaBerge thesis", 2024, "reproduction audit", "random vs patient-wise", "Random splitting dramatically underestimates error", "patient separation and duplicate audit"],
        ["PaPaGei", 2025, "PPG foundation ResNet1D-MoE", "task dependent", "Strong broad PPG representation; BP/device invariance not guaranteed", "frozen and fine-tuned foundation baselines"],
        ["Pulse-PPG", 2025, "field-trained PPG encoder", "cross-context", "Robust raw PPG representations, not a direct BP solution", "device-robust pretraining is promising"],
        ["Chu et al.", 2023, "MLM Transformer", "patient-wise train/val/test", "Fine-tuning may be patient-specific", "self-supervision and multi-task targets"],
        ["PulseDB", 2023, "curated MIMIC/VitalDB benchmark", "explicit subject IDs", "High-quality selection can narrow real-world domain", "calibration-free protocol and QC"],
        ["PPG-BP database", 2018, "219 subjects; 3x2.1 s PPG", "subject unit", "Cuff label is not simultaneous invasive ABP", "small independent device stress test"],
        ["Cross-dataset transfer study", 2026, "cross-corpus transfer", "external datasets", "Dataset/device shift dominates architecture gains", "zero-shot plus calibrated transfer"],
        ["Direct vs ECG-mediated", 2026, "BiLSTM direct/mediated", "large preprint corpus", "Very low reported errors require split scrutiny", "direct PPG path and explicit split audit"],
    ]
    columns = ["paper_or_resource", "year", "method", "evaluation", "main_gap", "adopted_test"]
    pd.DataFrame(rows, columns=columns).to_csv(TABLE / "literature_gap_matrix.csv", index=False)


def build_markdown(search, cv, oof, external, calibration, foundation):
    best_validation = search[search.selection_score.notna()].sort_values("selection_score").iloc[0]
    ensemble = oof[oof.model == "simple_mean_all_3"].iloc[0]
    winner = cv.sort_values("selection_score_mean").iloc[0]
    ppgbp = external[external.dataset == "ppgbp219"].iloc[0]
    papagei_cal = calibration[(calibration.dataset_model == "papagei10s_ambulatory56") & (calibration.method == "affine_ridge")].iloc[0]
    foundation_best = foundation.sort_values("selection_score").iloc[0]
    result = f"""# گزارش نهایی فاز دوم: جست‌وجوی ماژولار PPG به فشار خون

## جمع‌بندی اجرایی

بهترین مدل منفردِ تأییدشده یک ResNet-1D با ورودی چندنمایی ۱۰ ثانیه‌ای (PPG، VPG، APG، مؤلفهٔ هموار و مؤلفهٔ detrended) و سر multitask برای SBP/DBP/MAP/PP بود. میانگین ۱۵ ارزیابی patient-wise آن `SBP={winner.mae_sbp_mean:.2f}±{winner.mae_sbp_sd:.2f}` و `DBP={winner.mae_dbp_mean:.2f}±{winner.mae_dbp_sd:.2f} mmHg` شد.

ensemble مساوی سه finalist، روی پیش‌بینی OOF تجمیع‌شدهٔ سه seed، به `SBP={ensemble.mae_sbp:.2f}` و `DBP={ensemble.mae_dbp:.2f} mmHg` رسید؛ bootstrap خوشه‌ای ۹۵٪ به‌ترتیب `{ensemble.mae_sbp_ci95}` و `{ensemble.mae_dbp_ci95}` بود. این بهترین روش داخلی پروژه است، ولی هنوز معیار بالینی AAMI را برآورده نمی‌کند.

در external zero-shot افت شدید دیده شد: مدل ۲ثانیه‌ای روی PPG-BP به `{ppgbp.mae_sbp:.2f}/{ppgbp.mae_dbp:.2f}` رسید. PaPaGei frozen نیز zero-shot موفق نبود، اما نگاشت خطی کاملاً cross-fitted روی دیتاست ۵۶نفره آن را به `{papagei_cal.mae_sbp:.2f}/{papagei_cal.mae_dbp:.2f}` رساند. بنابراین پیام علمی اصلی «نیاز به device/domain calibration» است، نه ادعای calibration-free بالینی.

## دامنهٔ آزمایش

- ۱۹۸ مسیر PDF کشف شد؛ ۱۱۵ PDF بایت‌به‌بایت یکتا بود؛ برای هر ۱۱۵ مورد full-text استخراج شد و ۸۷ evidence card مرتبط ساخته شد.
- ۶۵ پیکربندی deep/foundation در جست‌وجوی قفل‌شده اجرا شد: ۱۷ معماری، ۲۴ ترکیب ورودی/preprocessing و ۲۴ ترکیب head/loss/augmentation.
- پنج fine-tuning بنیادین exploratory افزوده شد.
- سه finalist در ۵ fold و ۳ seed، در مجموع ۴۵ fit تأییدی، با بیمارهای کاملاً جدا ارزیابی شدند.
- برای ارزیابی خارجی، ensembleهای ۱۵مدلی روی دو دیتاست مستقل ساخته شد.
- test قدیمی فاز اول قبلاً دیده شده بود؛ فاز دوم آن را untouched test ننامید. screening فقط train/validation قدیمی را دید و گزارش اصلی از repeated grouped OOF و external stress test می‌آید.

## نتایج بلوک‌ها

بهترین validation تک‌اجرا `{best_validation.run_id}` با `MAE={best_validation.mae_sbp:.2f}/{best_validation.mae_dbp:.2f}` بود. بااین‌حال winner نهایی از میانگین ۱۵ fold/seed انتخاب شد، نه از این تک‌عدد.

- backbone: ResNet تخصصی از FCN، TCN، gated-TCN، InceptionTime، ConvNeXt-1D، BiGRU، Patch Transformer، Patch Mixer، multi-scale، spectral fusion و PaPaGei بهتر بود.
- duration/view: پنج ثانیه derivative و ده ثانیه multiview بهترین توازن را داشتند؛ ۱۵ ثانیه الزاماً بهتر نبود و دو ثانیه افت داشت.
- objective: multitask فیزیولوژیک MAP/PP همراه jitter/scale در validation بهترین بود؛ Gaussian NLL و tail oversampling بهتر نشدند.
- foundation: بهترین fine-tuning exploratory PaPaGei `{foundation_best.run_id}` با `{foundation_best.mae_sbp:.2f}/{foundation_best.mae_dbp:.2f}` بود و مدل تخصصی را شکست نداد.
- ensemble: وزن مساوی از weight tuning پس‌نگرانه قابل‌دفاع‌تر است و اندکی هر سه finalist را بهتر کرد.

## نتیجهٔ قابل‌مقاله

موضوع پیشنهادی نهایی: **Leakage-Aware Component-Wise PPG-to-BP Estimation: Multiview Physiological Multitasking and Cross-Device Stress Testing**.

نوآوری قابل‌دفاع در کنار هم قرار گرفتن چهار جزء است: جست‌وجوی component-wise به‌جای معرفی یک شبکهٔ مبهم؛ قفل‌کردن انتخاب در سطح بیمار؛ مقایسهٔ مستقیم foundation PPG با مدل تخصصی؛ و نشان‌دادن تفاوت چشمگیر within-domain، zero-shot و locally calibrated transfer روی دو دستگاه خارجی.

## محدودیت‌های الزامی برای مقاله

- دادهٔ داخلی ICU و برچسب ABP با دیتاست‌های خارجی cuff و غیرهمزمان یکسان نیست.
- split تست فاز اول قبلاً افشا شده و نباید در عنوان یا متن «untouched» خوانده شود.
- مقایسه با اعداد ۱ تا ۵ mmHg مقالات random-segment منصفانه نیست.
- calibration خارجی supervised است و باید جدا از zero-shot گزارش شود.
- نتیجهٔ PPG-BP با مدل ۲ثانیه‌ای گزارش شده؛ مدل ۱۰ثانیه‌ای روی سیگنال تکرارشده مبنای ادعا نیست.

## فایل‌های شکل

- `figures/01_stage_a_architectures.png`
- `figures/02_duration_view_ablation.png`
- `figures/03_objective_ablation.png`
- `figures/04_confirmation_oof.png`
- `figures/05_bland_altman_oof.png`
- `figures/06_external_transfer.png`
"""
    (OUT / "PHASE2_RESULTS_FA.md").write_text(result, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    search = pd.read_csv(P2 / "outputs/search_results.csv")
    cv = pd.read_csv(P2 / "outputs/confirmation_cv/summary.csv")
    oof = pd.read_csv(P2 / "outputs/analysis/oof_ensemble_metrics.csv")
    external = pd.read_csv(P2 / "outputs/external_validation/external_metrics.csv")
    calibration = pd.read_csv(P2 / "outputs/external_validation/calibration/crossfit_calibration_metrics.csv")
    foundation = pd.read_csv(P2 / "outputs/foundation_adaptation/results.csv")
    search.to_csv(TABLE / "all_65_search_runs.csv", index=False)
    cv.to_csv(TABLE / "confirmation_summary.csv", index=False)
    oof.to_csv(TABLE / "oof_metrics.csv", index=False)
    external.to_csv(TABLE / "external_zero_shot.csv", index=False)
    calibration.to_csv(TABLE / "external_crossfit_calibration.csv", index=False)
    architecture_figure(search)
    block_figure(search)
    objective_figure(search)
    internal_figure(oof)
    bland_altman()
    external_figure(calibration)
    literature_matrix()
    build_markdown(search, cv, oof, external, calibration, foundation)
    print(OUT / "PHASE2_RESULTS_FA.md")


if __name__ == "__main__":
    main()

