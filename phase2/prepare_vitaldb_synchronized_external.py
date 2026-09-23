from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "phase2/protocol/VITALDB_SYNCHRONIZED_EXTERNAL_LOCK.json"
DATA = ROOT / "data/external/vitaldb"
RAW = DATA / "raw"
PREPARED = DATA / "prepared"
API = "https://api.vitaldb.net"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 4) -> None:
    if destination.exists() and destination.stat().st_size > 100:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ICBME2026-research-audit/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1024 * 1024)
            if part.stat().st_size <= 100:
                raise RuntimeError(f"Downloaded payload is too small: {part.stat().st_size}")
            part.replace(destination)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def load_api_table(endpoint: str, destination: Path) -> pd.DataFrame:
    download(f"{API}/{endpoint}", destination)
    with gzip.open(destination, "rb") as handle:
        return pd.read_csv(handle)


def deterministic_cases(tracks: pd.DataFrame, count: int) -> tuple[list[int], pd.DataFrame]:
    required = ["SNUADC/PLETH", "Solar8000/ART_SBP", "Solar8000/ART_DBP"]
    eligible = set.intersection(*(set(tracks.loc[tracks.tname == name, "caseid"].astype(int)) for name in required))
    ranked = sorted(eligible, key=lambda case: hashlib.sha256(f"ICBME2026-vitaldb-sync-v1:{case}".encode()).hexdigest())
    selected = ranked[:count]
    rows = tracks[tracks.caseid.isin(selected) & tracks.tname.isin(required)].copy()
    if rows.groupby("caseid").tname.nunique().min() != 3 or len(selected) != count:
        raise RuntimeError("Incomplete deterministic track selection")
    return selected, rows


def read_waveform(path: Path) -> tuple[float, float, np.ndarray]:
    values: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        first = next(reader)
        second = next(reader)
        start = float(first[0])
        interval = float(second[0])
        for row in reader:
            if not row:
                continue
            if row[0].strip():
                continue
            try:
                values.append(float(row[1]))
            except (IndexError, ValueError):
                values.append(np.nan)
    return start, interval, np.asarray(values, dtype=np.float32)


def read_numeric(path: Path) -> tuple[np.ndarray, np.ndarray]:
    time_values, measurements = [], []
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            try:
                t, value = float(row[0]), float(row[1])
            except (IndexError, ValueError):
                continue
            if np.isfinite(t) and np.isfinite(value):
                time_values.append(t)
                measurements.append(value)
    return np.asarray(time_values), np.asarray(measurements)


def standardize(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    if mad < 1e-5:
        mad = max(float(np.std(values)), 1e-5)
    return np.clip((values - median) / mad, -8, 8).astype(np.float32)


def robust_channels(raw: np.ndarray) -> np.ndarray:
    center = np.median(raw)
    scale = max(float(np.quantile(raw, 0.75) - np.quantile(raw, 0.25)), 1e-5)
    ppg = np.clip((raw - center) / scale, -10, 10).astype(np.float32)
    vpg = np.diff(ppg, prepend=ppg[:1])
    apg = np.diff(vpg, prepend=vpg[:1])
    return np.stack([ppg, standardize(vpg), standardize(apg)])


def ppg_quality(values: np.ndarray, fs: int = 125) -> tuple[bool, dict]:
    finite = np.isfinite(values)
    finite_fraction = float(finite.mean())
    if finite_fraction < 0.95:
        return False, {"finite_fraction": finite_fraction}
    index = np.arange(len(values))
    signal = np.interp(index, index[finite], values[finite])
    iqr = float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25))
    if iqr < 1e-5:
        return False, {"finite_fraction": finite_fraction, "iqr": iqr}
    z = (signal - np.median(signal)) / iqr
    spectrum = np.abs(np.fft.rfft(z - z.mean())) ** 2
    frequency = np.fft.rfftfreq(len(z), 1 / fs)
    pulse = (frequency >= 0.5) & (frequency <= 3.5)
    total = (frequency >= 0.2) & (frequency <= 10)
    ratio = float(spectrum[pulse].sum() / max(spectrum[total].sum(), 1e-12))
    peaks, _ = find_peaks(z, distance=int(0.3 * fs), prominence=0.15)
    count = int(len(peaks))
    passed = ratio >= 0.25 and 6 <= count <= 40
    return passed, {"finite_fraction": finite_fraction, "iqr": iqr, "pulse_band_fraction": ratio, "pulse_count": count, "interpolated": signal}


def prepare_case(case: int, track_rows: pd.DataFrame) -> tuple[list[np.ndarray], list[dict], dict]:
    lookup = dict(zip(track_rows.tname, track_rows.tid))
    paths = {name: RAW / f"case{case:04d}_{name.split('/')[-1]}.csv.gz" for name in lookup}
    start, interval, waveform = read_waveform(paths["SNUADC/PLETH"])
    if not np.isclose(interval, 0.002, atol=1e-6):
        raise RuntimeError(f"Unexpected PLETH interval for case {case}: {interval}")
    waveform = waveform[::4]
    fs = 125
    ppg_start = start
    sbp_t, sbp_v = read_numeric(paths["Solar8000/ART_SBP"])
    dbp_t, dbp_v = read_numeric(paths["Solar8000/ART_DBP"])
    stop = ppg_start + len(waveform) / fs
    candidate_rows: list[tuple[np.ndarray, dict]] = []
    rejections: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    for window_start in np.arange(math.ceil(ppg_start / 15) * 15, stop - 15, 15):
        sample_start = int(round((window_start - ppg_start) * fs))
        raw = waveform[sample_start: sample_start + 15 * fs]
        if len(raw) != 15 * fs:
            reject("short_ppg")
            continue
        passed, quality = ppg_quality(raw, fs)
        if not passed:
            reject("ppg_qc")
            continue
        smask = (sbp_t >= window_start) & (sbp_t < window_start + 15)
        dmask = (dbp_t >= window_start) & (dbp_t < window_start + 15)
        if smask.sum() < 5 or dmask.sum() < 5:
            reject("insufficient_art")
            continue
        sbp = float(np.median(sbp_v[smask]))
        dbp = float(np.median(dbp_v[dmask]))
        sbp_iqr = float(np.quantile(sbp_v[smask], 0.75) - np.quantile(sbp_v[smask], 0.25))
        dbp_iqr = float(np.quantile(dbp_v[dmask], 0.75) - np.quantile(dbp_v[dmask], 0.25))
        if not (55 <= sbp <= 220 and 25 <= dbp <= 130 and sbp - dbp >= 10):
            reject("art_range")
            continue
        if sbp_iqr > 20 or dbp_iqr > 15:
            reject("art_instability")
            continue
        signal = quality.pop("interpolated")
        candidate_rows.append((robust_channels(signal), {
            "caseid": case,
            "window_start_seconds": float(window_start),
            "sbp": sbp,
            "dbp": dbp,
            "sbp_iqr": sbp_iqr,
            "dbp_iqr": dbp_iqr,
            **quality,
        }))

    if len(candidate_rows) > 20:
        chosen = np.unique(np.rint(np.linspace(0, len(candidate_rows) - 1, 20)).astype(int))
        candidate_rows = [candidate_rows[i] for i in chosen]
    return [row[0] for row in candidate_rows], [row[1] for row in candidate_rows], {
        "caseid": case,
        "raw_duration_seconds": stop - ppg_start,
        "accepted_windows": len(candidate_rows),
        "rejections": rejections,
    }


def main() -> None:
    protocol = json.loads(LOCK.read_text(encoding="utf-8"))
    if not protocol["locked_before_downloading_waveform_outcomes"]:
        raise RuntimeError("Protocol is not locked")
    RAW.mkdir(parents=True, exist_ok=True)
    PREPARED.mkdir(parents=True, exist_ok=True)
    tracks = load_api_table("trks", RAW / "trks.csv.gz")
    cases_table = load_api_table("cases", RAW / "cases.csv.gz")
    selected, rows = deterministic_cases(tracks, int(protocol["selection"]["requested_cases"]))
    rows.to_csv(PREPARED / "SELECTED_TRACKS.csv", index=False)
    cases_table[cases_table.caseid.isin(selected)].to_csv(PREPARED / "SELECTED_CASES.csv", index=False)

    jobs = []
    for row in rows.itertuples(index=False):
        destination = RAW / f"case{int(row.caseid):04d}_{row.tname.split('/')[-1]}.csv.gz"
        jobs.append((f"{API}/{row.tid}", destination))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download, url, destination): destination for url, destination in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            future.result()
            if number % 32 == 0 or number == len(futures):
                print(f"downloaded_or_verified={number}/{len(futures)}", flush=True)

    signals, metadata, case_audit = [], [], []
    by_case = {case: rows[rows.caseid == case] for case in selected}
    for number, case in enumerate(selected, 1):
        local_signals, local_metadata, audit = prepare_case(case, by_case[case])
        signals.extend(local_signals)
        metadata.extend(local_metadata)
        case_audit.append(audit)
        print(f"prepared={number}/{len(selected)} case={case} accepted={audit['accepted_windows']}", flush=True)
    if not signals:
        raise RuntimeError("No VitalDB windows passed QC")
    x = np.asarray(signals, dtype=np.float16)
    meta = pd.DataFrame(metadata)
    output = PREPARED / "vitaldb_sync128.npz"
    np.savez_compressed(
        output,
        x=x,
        y=meta[["sbp", "dbp"]].to_numpy(np.float32),
        patient_id=meta.caseid.to_numpy(np.int32),
        window_start_seconds=meta.window_start_seconds.to_numpy(np.float32),
        fs=np.asarray(125),
    )
    meta.to_csv(PREPARED / "WINDOW_METADATA.csv", index=False)
    (PREPARED / "CASE_QC_AUDIT.json").write_text(json.dumps(case_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accepted_cases = int(meta.caseid.nunique())
    success = accepted_cases >= 80 and len(meta) >= 800
    manifest = {
        "status": "PASS" if success else "FAIL",
        "selected_cases": len(selected),
        "accepted_cases": accepted_cases,
        "accepted_windows": int(len(meta)),
        "shape": list(x.shape),
        "sbp_range": [float(meta.sbp.min()), float(meta.sbp.max())],
        "dbp_range": [float(meta.dbp.min()), float(meta.dbp.max())],
        "high_bp_prevalence": float(((meta.sbp >= 140) | (meta.dbp >= 90)).mean()),
        "prepared_sha256": sha256(output),
        "downloaded_bytes": int(sum(path.stat().st_size for _, path in jobs)),
        "selection_rule": protocol["selection"]["method"],
    }
    (PREPARED / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
