#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs/CODE_MANIFEST.json"
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".tex", ".cff"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts or path == DESTINATION:
            continue
        if path.suffix.lower() not in SUFFIXES and path.name not in {"requirements.txt", "LICENSE"}:
            continue
        files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": len(files), "files": files,
        "note": "Generated source inventory; data integrity is recorded separately in data/DATA_MANIFEST.json.",
    }
    DESTINATION.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{DESTINATION}: {len(files)} source/configuration files")


if __name__ == "__main__":
    main()
