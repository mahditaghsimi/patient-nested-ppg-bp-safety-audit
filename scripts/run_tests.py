#!/usr/bin/env python3
"""Dependency-free runner for the repository's lightweight contract tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    path = ROOT / "tests/test_contracts.py"
    spec = importlib.util.spec_from_file_location("test_contracts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = sorted(name for name in vars(module) if name.startswith("test_") and callable(getattr(module, name)))
    for name in tests:
        getattr(module, name)()
        print(f"PASS {name}")
    print(f"PASS: {len(tests)} contract tests")


if __name__ == "__main__":
    main()
