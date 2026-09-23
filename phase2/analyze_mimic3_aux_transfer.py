from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase2/outputs/mimic3_aux_transfer_screen"


def summarize(files: list[Path], policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    pairs = []
    for (model, seed), group in data.groupby(["model", "seed"]):
        if set(group["mode"]) != {"scratch", "supervised_aux_pretrain"}:
            continue
        scratch = group[group["mode"] == "scratch"].iloc[0]
        transfer = group[group["mode"] == "supervised_aux_pretrain"].iloc[0]
        pairs.append(
            {
                "policy": policy,
                "model": model,
                "seed": int(seed),
                "scratch_sbp_mae": scratch.sbp_mae,
                "pretrained_sbp_mae": transfer.sbp_mae,
                "delta_sbp_mae": transfer.sbp_mae - scratch.sbp_mae,
                "scratch_dbp_mae": scratch.dbp_mae,
                "pretrained_dbp_mae": transfer.dbp_mae,
                "delta_dbp_mae": transfer.dbp_mae - scratch.dbp_mae,
                "scratch_mean_mae": scratch.mean_mae,
                "pretrained_mean_mae": transfer.mean_mae,
                "delta_mean_mae": transfer.mean_mae - scratch.mean_mae,
            }
        )
    pair_table = pd.DataFrame(pairs)
    summaries = []
    for model, group in pair_table.groupby("model"):
        delta = group.delta_mean_mae.to_numpy()
        n = len(delta)
        if n > 1:
            sem = stats.sem(delta)
            ci = stats.t.interval(0.95, n - 1, loc=np.mean(delta), scale=sem)
        else:
            ci = (np.nan, np.nan)
        summaries.append(
            {
                "policy": policy,
                "model": model,
                "seeds": n,
                "scratch_sbp_mae_mean": group.scratch_sbp_mae.mean(),
                "pretrained_sbp_mae_mean": group.pretrained_sbp_mae.mean(),
                "delta_sbp_mae_mean": group.delta_sbp_mae.mean(),
                "delta_sbp_mae_sd": group.delta_sbp_mae.std(ddof=1),
                "scratch_dbp_mae_mean": group.scratch_dbp_mae.mean(),
                "pretrained_dbp_mae_mean": group.pretrained_dbp_mae.mean(),
                "delta_dbp_mae_mean": group.delta_dbp_mae.mean(),
                "delta_dbp_mae_sd": group.delta_dbp_mae.std(ddof=1),
                "scratch_mean_mae_mean": group.scratch_mean_mae.mean(),
                "pretrained_mean_mae_mean": group.pretrained_mean_mae.mean(),
                "delta_mean_mae_mean": group.delta_mean_mae.mean(),
                "delta_mean_mae_sd": group.delta_mean_mae.std(ddof=1),
                "delta_mean_mae_ci95_low": ci[0],
                "delta_mean_mae_ci95_high": ci[1],
                "improved_seed_count": int((delta < 0).sum()),
                "all_seeds_improved": bool(np.all(delta < 0)),
            }
        )
    return pair_table, pd.DataFrame(summaries)


def main() -> None:
    global_files = sorted(path for path in OUT.glob("results_seed*.csv") if "smoke" not in path.name)
    fold_files = sorted(path for path in OUT.glob("results_fold_safe_seed*.csv") if "smoke" not in path.name)
    global_pairs, global_summary = summarize(global_files, "global_nonoverlap")
    fold_pairs, fold_summary = summarize(fold_files, "fold_safe")
    pairs = pd.concat([global_pairs, fold_pairs], ignore_index=True)
    summary = pd.concat([global_summary, fold_summary], ignore_index=True)
    # Frozen PaPaGei had no trainable transferred representation after head reset.
    summary["valid_pretraining_comparison"] = summary.model != "papagei_frozen"
    summary["interpretation"] = np.where(
        summary.model == "papagei_frozen",
        "invalid_as_pretraining_effect_frozen_backbone_and_reset_head; initialization_control_only",
        np.where(
            (summary.delta_mean_mae_ci95_high < 0) & (summary.delta_mean_mae_mean < -0.1),
            "potentially_meaningful_improvement_requires_confirmation",
            "no_confirmed_meaningful_improvement",
        ),
    )
    pairs.to_csv(OUT / "MULTISEED_PAIRED_DELTAS.csv", index=False)
    summary.to_csv(OUT / "MULTISEED_SUMMARY.csv", index=False)

    valid = summary[summary.valid_pretraining_comparison].sort_values("delta_mean_mae_mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = valid.policy + " / " + valid.model
    ax.barh(labels, valid.delta_mean_mae_mean, xerr=valid.delta_mean_mae_sd.fillna(0), color=np.where(valid.delta_mean_mae_mean < 0, "tab:green", "tab:red"))
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Pretrained minus scratch mean MAE (mmHg); negative is better")
    ax.set_title("Auxiliary pretraining effect across seeds")
    fig.tight_layout()
    fig.savefig(OUT / "multiseed_pretraining_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = f"""# نتیجهٔ چند-seed انتقال کمکی MIMIC-III

مقدار منفی delta به معنی بهترشدن pretraining نسبت به scratch است.

```
{valid[['policy','model','seeds','delta_sbp_mae_mean','delta_dbp_mae_mean','delta_mean_mae_mean','delta_mean_mae_sd','improved_seed_count','interpretation']].to_string(index=False)}
```

## تصمیم

- هیچ روش معتبر فعلی بهبود میانگین حداقل 0.1 mmHg با CI کاملاً زیر صفر نشان نداد.
- اثر ResNet بین seedها ناپایدار است.
- spectral fusion بهبود کوچک دارد، اما اندازهٔ اثر برای ادعای مقاله کافی نیست.
- PaPaGei last-block در fold-safe از نظر delta سازگارتر است، ولی MAE مطلق آن از ResNet ضعیف‌تر است.
- PaPaGei frozen از تحلیل اثر pretraining حذف شد، چون backbone منجمد و head بازنشانی شده بود؛ اختلاف آن صرفاً کنترل initialization است.

نتیجهٔ علمی فعلی منفی/خنثی است: pretraining supervised ساده روی دادهٔ کمکی leakage-safe، بهبود
معنادار و قابل اتکایی نسبت به scratch ایجاد نکرده است. راهبردهای بعدی باید از نظر انتقال representation
واقعاً متفاوت باشند و بدون بهبود transition/external نباید به‌عنوان نوآوری معرفی شوند.
"""
    (OUT / "MULTISEED_RESULTS_FA.md").write_text(report, encoding="utf-8")
    print(valid.to_string(index=False))


if __name__ == "__main__":
    main()
