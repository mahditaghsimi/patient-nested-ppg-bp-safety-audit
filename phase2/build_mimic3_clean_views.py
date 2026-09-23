from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMPORT = ROOT / "data/external/mimic3/imports/mimic3_server_bundle_20260820"
DATA = IMPORT / "extracted/phase2/outputs/mimic3_dirty_aware_3gb"
MAPPING_SOURCE = IMPORT / "mapping_evidence/MIMIC3WDB_MATCHED_SHA256SUMS.txt"
LOCAL_LABELS = ROOT / "data/mimic_bp/raw/labels"
OUT = ROOT / "phase2/outputs/mimic3_clean_views"
PATH_RE = re.compile(r"(?:^|/)(?P<subject>p\d{6})/(?P<file>3\d{6}(?:_\d{4})?\.(?:hea|dat))$")
EXPECTED_MAPPING_BYTES = 172_636_445


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_mapping(accepted: pd.DataFrame) -> pd.DataFrame:
    if not MAPPING_SOURCE.exists():
        raise FileNotFoundError(f"Official mapping evidence is missing: {MAPPING_SOURCE}")
    if MAPPING_SOURCE.stat().st_size != EXPECTED_MAPPING_BYTES:
        raise RuntimeError(
            f"Official mapping evidence is incomplete: {MAPPING_SOURCE.stat().st_size} != {EXPECTED_MAPPING_BYTES} bytes"
        )
    wanted_segments = set(accepted.segment.astype(str))
    wanted_records = {value.split("/")[-1] for value in accepted.record.astype(str)}
    segment_subjects: dict[str, set[str]] = defaultdict(set)
    record_subjects: dict[str, set[str]] = defaultdict(set)
    with MAPPING_SOURCE.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            path = line.split(maxsplit=1)[-1].strip().lstrip("*").replace("\\", "/")
            match = PATH_RE.search(path)
            if not match:
                continue
            stem = match.group("file").rsplit(".", 1)[0]
            base_record = stem.split("_", 1)[0]
            if stem not in wanted_segments and base_record not in wanted_records:
                continue
            subject = match.group("subject")[1:]
            if stem in wanted_segments:
                segment_subjects[stem].add(subject)
            if base_record in wanted_records:
                record_subjects[base_record].add(subject)

    rows = []
    for item in accepted[["record", "segment"]].drop_duplicates().itertuples(index=False):
        base_record = item.record.split("/")[-1]
        subjects = segment_subjects.get(item.segment, set()) or record_subjects.get(base_record, set())
        rows.append(
            {
                "record": item.record,
                "segment": item.segment,
                "official_subject_id": next(iter(subjects)) if len(subjects) == 1 else "|".join(sorted(subjects)),
                "mapping_status": "official_unique"
                if len(subjects) == 1
                else "official_ambiguous"
                if len(subjects) > 1
                else "unmatched_record_proxy_only",
                "mapping_candidate_count": len(subjects),
            }
        )
    return pd.DataFrame(rows)


def fill_1d(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if finite.all():
        return x
    result = x.copy()
    index = np.arange(len(x))
    result[~finite] = np.interp(index[~finite], index[finite], x[finite])
    return result


def waveform_flags(accepted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in sorted((DATA / "shards").glob("windows_*.npz")):
        with np.load(path, allow_pickle=False) as source:
            ppg = source["ppg"].astype(np.float32)
        for shard_row, signal in enumerate(ppg):
            missing = int((~np.isfinite(signal)).sum())
            filled = fill_1d(signal) if missing else signal
            low, high = np.percentile(filled, [1, 99])
            robust_range = float(high - low)
            differences = np.abs(np.diff(filled))
            median_difference = float(np.median(differences)) if len(differences) else 0.0
            max_difference = float(np.max(differences)) if len(differences) else 0.0
            step = bool(max_difference > max(20 * median_difference, 0.75 * robust_range) and max_difference > 1e-6)
            rows.append(
                {
                    "shard": f"shards/{path.name}",
                    "shard_row": shard_row,
                    "missing_samples": missing,
                    "missing_fraction": missing / len(signal),
                    "finite": missing == 0,
                    "step_artifact": step,
                    "interpolation_policy": "linear_between_nearest_finite_samples" if missing else "not_needed",
                }
            )
    flags = pd.DataFrame(rows)
    if len(flags) != len(accepted):
        raise RuntimeError(f"Shard row mismatch: {len(flags)} != {len(accepted)}")
    return flags


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    accepted = pd.read_csv(DATA / "ACCEPTED_WINDOWS.csv")
    mapping = full_mapping(accepted)
    flags = waveform_flags(accepted)
    view = accepted.merge(flags, on=["shard", "shard_row"], validate="one_to_one")
    view = view.merge(mapping, on=["record", "segment"], how="left", validate="many_to_one")
    local_subjects = {path.name[1:7] for path in LOCAL_LABELS.glob("p*_labels.npy")}
    view["known_mimic_bp_overlap"] = (view.mapping_status == "official_unique") & view.official_subject_id.isin(local_subjects)
    view["known_nonoverlap"] = (view.mapping_status == "official_unique") & ~view.official_subject_id.isin(local_subjects)
    view["mapping_unknown"] = view.mapping_status != "official_unique"
    view["view_strict_finite"] = view.finite
    view["view_interpolated"] = view.missing_fraction <= 0.01
    view["view_high_confidence"] = (
        view.finite
        & ~view.step_artifact
        & (view.lag_correlation.abs() >= 0.5)
        & (view.ppg_sqi >= 0.5)
        & (view.abp_sqi >= 0.5)
    )
    view["eligible_for_confirmatory_merge"] = view.view_high_confidence & view.known_nonoverlap
    view["eligible_for_exploratory_pretraining"] = view.view_interpolated

    mapping.to_csv(OUT / "OFFICIAL_MAPPING_ALL_ACCEPTED_RECORDS.csv", index=False)
    view.to_csv(OUT / "MIMIC3_CLEAN_VIEW_INDEX.csv", index=False)
    statuses = mapping.mapping_status.value_counts().to_dict()
    summary = {
        "status": "completed",
        "source": str(DATA),
        "accepted_windows": int(len(view)),
        "record_groups": int(view.record.nunique()),
        "mapping_status_records": {str(k): int(v) for k, v in statuses.items()},
        "official_unique_subjects": int(mapping.loc[mapping.mapping_status == "official_unique", "official_subject_id"].nunique()),
        "known_overlap_subjects": int(view.loc[view.known_mimic_bp_overlap, "official_subject_id"].nunique()),
        "known_overlap_records": int(view.loc[view.known_mimic_bp_overlap, "record"].nunique()),
        "known_overlap_windows": int(view.known_mimic_bp_overlap.sum()),
        "known_nonoverlap_subjects": int(view.loc[view.known_nonoverlap, "official_subject_id"].nunique()),
        "known_nonoverlap_windows": int(view.known_nonoverlap.sum()),
        "mapping_unknown_windows": int(view.mapping_unknown.sum()),
        "views": {
            "strict_finite": int(view.view_strict_finite.sum()),
            "interpolated": int(view.view_interpolated.sum()),
            "high_confidence": int(view.view_high_confidence.sum()),
            "confirmatory_mapped_nonoverlap": int(view.eligible_for_confirmatory_merge.sum()),
        },
        "quality": {
            "nan_windows": int((~view.finite).sum()),
            "step_artifact_windows": int(view.step_artifact.sum()),
            "abs_lag_correlation_below_0_5": int((view.lag_correlation.abs() < 0.5).sum()),
            "ppg_sqi_below_0_5": int((view.ppg_sqi < 0.5).sum()),
            "abp_sqi_below_0_5": int((view.abp_sqi < 0.5).sum()),
        },
        "hashes": {
            "mapping_source_sha256": sha256(MAPPING_SOURCE),
            "accepted_csv_sha256": sha256(DATA / "ACCEPTED_WINDOWS.csv"),
        },
        "policy": {
            "raw_shards_modified": False,
            "unknown_mapping_is_not_nonoverlap": True,
            "random_window_split_allowed": False,
            "primary_confirmatory_view": "high_confidence AND official_unique AND known_nonoverlap",
            "exploratory_pretraining_view": "interpolated; report overlap limitation and use fold-specific exclusions",
        },
    }
    (OUT / "CLEAN_VIEW_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# نمای پاک و ممیزی هم‌پوشانی MIMIC-III

این خروجی shardهای واردشده را تغییر نمی‌دهد و فقط یک index مشتق‌شده می‌سازد.

## نگاشت

- record پذیرفته‌شده: {summary['record_groups']:,}
- subject رسمی یکتا: {summary['official_unique_subjects']:,}
- window با هم‌پوشانی قطعی MIMIC-BP: {summary['known_overlap_windows']:,}
- window با non-overlap رسمی: {summary['known_nonoverlap_windows']:,}
- window با mapping نامشخص: {summary['mapping_unknown_windows']:,}

## viewها

- strict finite: {summary['views']['strict_finite']:,}
- interpolated: {summary['views']['interpolated']:,}
- high confidence: {summary['views']['high_confidence']:,}
- confirmatory mapped non-overlap: {summary['views']['confirmatory_mapped_nonoverlap']:,}

`unknown` هرگز معادل `non-overlap` در نظر گرفته نشده است. ادغام تأییدی فقط برای ردیف‌های
high-confidence با subject رسمی و خارج از MIMIC-BP مجاز است. داده‌های mapping-unknown فقط
در pretraining اکتشافی و با محدودیت صریح قابل استفاده‌اند.
"""
    (OUT / "CLEAN_VIEWS_FA.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
