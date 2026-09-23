# بستهٔ تمیز FINAL برای اجرای مقالهٔ PPG به فشارخون

این پوشه نسخهٔ مستقل و مرتب pipeline پژوهش است. هیچ نتیجهٔ قدیمی به‌عنوان خروجی این نسخه کپی نشده است؛ بنابراین همهٔ خروجی‌ها باید روی سیستم مقصد از نو ساخته شوند.

## اجرای کامل

پس از قرار دادن داده‌ها طبق `data/README_FA.md` و نصب وابستگی‌ها:

```bash
python main.py
```

اجرای پیش‌فرض تمام مراحل preprocessing، آموزش، انتخاب مدل، nested evaluation، ارزیابی خارجی، تحلیل ایمنی، جدول‌ها، نمودارها، مقاله، تست‌ها و ممیزی نهایی را به‌ترتیب اجرا می‌کند. خروجی هر مرحله هم‌زمان در ترمینال چاپ و در `logs/<stage>.log` ذخیره می‌شود.

حالت‌های دیگر:

```bash
python main.py --list-stages
python main.py --dry-run
python main.py --mode compute
python main.py --mode report
python main.py --mode audit
python main.py --from-stage innovation_search
python main.py --force vitaldb_external
```

`--dry-run` فقط ترتیب دستورات را چاپ می‌کند. حالت عادی قابل‌ادامه است: مرحله فقط وقتی skip می‌شود که state معتبر، fingerprint یکسان کد و قرارداد خروجی معتبر داشته باشد. فایل state خراب یا صفر‌بایتی کنار گذاشته می‌شود و pipeline از حالت سالم ادامه می‌یابد.

## ساختار

- `main.py`: تنها نقطهٔ ورود.
- `pipeline/`: ترتیب مراحل، state، resume، logging و کنترل خروجی.
- `data/`: تمام ورودی‌های کاربر و داده‌های خارجی.
- `features/`: استخراج ویژگی‌های آماری، طیفی، مشتق و pulse.
- `neural/`: ساخت cache موج و مدل transfer اولیه.
- `phase2/`: مدل‌ها، جست‌وجوها، nested CV، safety و external validation.
- `scripts/`: ممیزی، ساخت جدول/شکل/مقاله و خلاصهٔ نهایی.
- `vendor/papagei/`: فقط کد upstream لازم برای PaPaGei؛ وزن آن باید در `data/pretraining/` قرار گیرد.
- `outputs/` و `phase2/outputs/`: checkpointها و نتایج محاسباتی.
- `results/tables/`: جدول‌های CSV مناسب مقاله.
- `results/figures/`: تمام نمودارهای PNG/PDF.
- `paper/`: LaTeX، PDF و DOCX تولیدشده.
- `artifacts/FINAL_SUMMARY.md`: خلاصهٔ چاپ‌شدهٔ نتایج نهایی.

## اصلاحات علمی و فنی این نسخه

- baseline خارجی اکنون فقط از partition رسمی train داخلی محاسبه می‌شود، نه از train+validation+test.
- مقاله صریحاً می‌گوید مدل خارجی، fixed prior ثانویه است و همان روش fold-varying nested نیست.
- خروجی‌های تاریخی کپی نشده‌اند تا با اجرای جدید مخلوط نشوند.
- claim audit اعداد تازهٔ مقاله را با artifactهای همان اجرا مقایسه می‌کند.
- مسیرهای شخصی Linux و محیط مجازی شخصی حذف شده‌اند.
- تمام داده‌ها زیر یک پوشهٔ `data/` قرار گرفته‌اند.
- خلاصهٔ نهایی، همهٔ اعداد اصلی و فهرست جدول‌ها و شکل‌ها را چاپ و ذخیره می‌کند.

## محیط پیشنهادی

- Python 3.11
- GPU دارای CUDA و فضای کافی برای بیش از هزار fit
- PyTorch سازگار با CUDA سیستم مقصد
- Tectonic روی `PATH` برای تولید PDF مقاله
- Poppler (`pdfinfo`) روی `PATH` برای کنترل PDF و تعداد صفحات
- Git LFS برای checkpointها و آرایه‌های بزرگ

ابتدا نسخهٔ CUDA مناسب PyTorch را نصب کنید، سپس:

```bash
python -m pip install -r requirements.txt
```

برای بازتولید نزدیک به محیط قبلی می‌توانید از `requirements-lock.txt` استفاده کنید، ولی نسخهٔ CUDA باید با driver سیستم مقصد سازگار باشد.

## نکات مهم مقاله

این pipeline برای مقالهٔ ممیزی روش‌شناختی و ایمنی مناسب است، نه برای ادعای دستگاه بالینی. پیش از ارسال مقاله باید نام نویسندگان، affiliation، statement اخلاق/IRB، مجوز داده‌ها و مشخصات کامل منابع تکمیل شوند.
