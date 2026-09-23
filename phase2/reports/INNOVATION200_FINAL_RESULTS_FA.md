# نتیجهٔ نهایی اجرای خودکار ۲۰۰ روش PPG→BP

تاریخ تولید: 2026-09-11T13:12:24.313731+00:00

## نتیجهٔ اصلی

- وضعیت تأیید: **NO_CONFIRMED_IMPROVEMENT**
- روش برنده: `PRIOR_WINNER_FIXED` — prior_resnet_multiview10_multitask
- OOF برنده، SBP/DBP MAE: **12.842/8.157 mmHg**.
- OOF مدل قبلی، SBP/DBP MAE: **12.842/8.157 mmHg**.
- fold-mean baseline، SBP/DBP MAE: **14.377/9.626 mmHg**.
- حساسیت مستقیم فشار بالا برای برنده: **3.36%**.

## کامل‌بودن اجرا

- screening موفق: **200/200**.
- confirmation بیمار-جدا: **75/75 fit**؛ ۵ fold × ۳ seed × 5 finalist.
- checkpoint ذخیره‌شده: **75**.
- زمان تجمعی screening: **30.4 دقیقه GPU**.

## بهترین نتیجهٔ screening

`I200_TRAIN_003_7a8882a0ec` / adamw_3e4_cos_amplitude_warp: 12.909/8.434 mmHg؛ Δmean نسبت به reference هم-seed = -0.173 mmHg.

## آزمون آماری بیمارمحور

هیچ روش جدیدی gate ازپیش‌تعریف‌شده را عبور نداد؛ مدل قبلی حفظ شد و نتیجهٔ منفی نیز مستند است.

## انتقال قفل‌شده به VitalDB

- کاندید external-compatible: `PRIOR_WINNER_FIXED`.
- zero-shot SBP/DBP MAE: **18.389/11.287 mmHg**.
- source-mean baseline: **17.846/10.448 mmHg**.
- حساسیت مستقیم فشار بالا: **0.00%**؛ AUPRC=0.312.

## تفسیر برای مقاله

- فقط وضعیت `CONFIRMED_IMPROVEMENT` مجوز ادعای بهبود مدل نسبت به مدل قبلی را می‌دهد.
- حتی در صورت بهبود داخلی، نتیجهٔ VitalDB باید مستقل و بدون پنهان‌کردن شکست انتقال گزارش شود.
- calibrated و zero-shot دو حالت متفاوت‌اند و نباید با هم ادغام شوند.
- grouped OOF پس از screening کاملاً nested نیست؛ این محدودیت باید در مقاله بماند.

## فایل‌های جزئی

- `phase2/outputs/innovation200_search/METHOD_BY_METHOD.csv`: هر ۲۰۰ روش.
- `phase2/outputs/innovation200_search/COMPONENT_EFFECTS.csv`: اثر هر جزء.
- `phase2/outputs/innovation200_confirmation/PAIRED_PATIENT_TESTS.csv`: آزمون paired.
- `phase2/outputs/innovation200_vitaldb_external/DECISION.json`: انتقال خارجی.
