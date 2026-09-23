from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "phase2"
OUT = P2 / "reports" / "ICBME2026_FINAL_SPRINT_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"No page count in pdfinfo output for {path}")


def csv_audit(path: Path, expected_rows: int) -> dict:
    frame = pd.read_csv(path)
    status = frame["status"].value_counts(dropna=False).to_dict()
    assert len(frame) == expected_rows, (path, len(frame), expected_rows)
    assert status == {"ok": expected_rows}, (path, status)
    return {"path": str(path.relative_to(ROOT)), "rows": len(frame), "status": status}


def main() -> None:
    paper = P2 / "paper" / "main_anonymous_medical.pdf"
    word = P2 / "paper" / "main_anonymous_medical_editable.docx"
    word_render = P2 / "paper" / "word_render_medical_v4" / "main_anonymous_medical_editable.pdf"
    tex = P2 / "paper" / "main_anonymous_medical.tex"

    for path in (paper, word, word_render, tex):
        assert path.exists() and path.stat().st_size > 0, path
    tex_text = tex.read_text(encoding="utf-8")
    assert "Clinical Safety Audit" in tex_text
    assert "Anonymous submission" in tex_text
    assert "TODO" not in tex_text and "PLACEHOLDER" not in tex_text
    assert pdf_pages(paper) == 4
    assert pdf_pages(word_render) == 4

    prior = pd.read_csv(P2 / "reports" / "tables" / "all_65_search_runs.csv")
    assert len(prior) == 65
    confirmation = pd.read_csv(P2 / "outputs" / "icbme_clinical_confirmation" / "fold_results.csv")
    assert len(confirmation) == 45
    assert confirmation["status"].value_counts().to_dict() == {"ok": 45}

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS",
        "screening": {
            "prior_component_search": {"rows": len(prior)},
            "new_clinical_search": csv_audit(
                P2 / "outputs" / "icbme_clinical_search" / "screening_results.csv", 96
            ),
            "new_event_search": csv_audit(
                P2 / "outputs" / "icbme_event_search" / "screening_results.csv", 36
            ),
            "total_structured_screening_configurations": 197,
        },
        "confirmation": {"fits": 45, "status": {"ok": 45}},
        "paper": {
            "anonymous_pdf": str(paper.relative_to(ROOT)),
            "anonymous_pdf_pages": pdf_pages(paper),
            "anonymous_pdf_sha256": sha256(paper),
            "editable_word": str(word.relative_to(ROOT)),
            "editable_word_sha256": sha256(word),
            "word_render_pages": pdf_pages(word_render),
            "word_render_sha256": sha256(word_render),
        },
        "remaining_manual_fields": [
            "author names and order",
            "affiliations and corresponding author",
            "at least one faculty author",
            "funding disclosure if applicable",
        ],
    }
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
