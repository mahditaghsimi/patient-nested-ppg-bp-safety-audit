# راهنمای قرار دادن داده‌ها

داده‌ها را دقیقاً با ساختار زیر قرار دهید. نام فایل‌های MIMIC-BP باید با شناسه‌های split رسمی تطابق داشته باشد.
فایل‌های split موجود در تحویل فعلی فقط placeholder صفر‌بایتی هستند و باید با نسخهٔ واقعی جایگزین شوند.

```text
data/
├── mimic_bp/
│   ├── raw/
│   │   ├── ppg/
│   │   │   └── pXXXX_ppg.npy       # 1524 فایل؛ هرکدام shape=(30, 3750)
│   │   └── labels/
│   │       └── pXXXX_labels.npy    # 1524 فایل؛ هرکدام shape=(30, 2)
│   └── splits/
│       ├── historical_custom_split_seed42.json  # فقط برای ممیزی crosswalk
│       └── official_source/
│           ├── train_subjects.txt
│           ├── val_subjects.txt
│           └── test_subjects.txt
├── pretraining/
│   ├── forecast_encoder_best_model.pt
│   └── papagei_s.pt
└── external/
    ├── ppg_bp/raw/Data File/
    │   ├── PPG-BP dataset.xlsx
    │   └── 0_subject/*.txt
    ├── ppg_ambulatory_56/raw/PPG_json/*.json
    ├── vitaldb/raw/                 # اختیاری؛ در صورت نبود، مرحلهٔ مربوط دانلود می‌کند
    ├── vitaldb/prepared/            # خودکار ساخته می‌شود
    └── prepared/                    # خودکار ساخته می‌شود
```

فایل‌های داخل `prepared/` را دستی نسازید. pipeline آن‌ها را همراه manifest و hash ایجاد می‌کند.

وجود داده در سیستم محلی به معنی اجازهٔ انتشار آن نیست. قبل از انتقال repository یا انتشار عمومی، شرایط MIMIC-BP، PPG-BP، مجموعهٔ ambulatory، VitalDB و checkpointهای pretrained را جداگانه بررسی کنید.
