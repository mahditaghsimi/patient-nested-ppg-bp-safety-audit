# Reproducibility protocol

## Hardware

Audit mode runs on CPU. Compute mode was designed for a CUDA GPU and contains 200 screening configurations, 75 secondary confirmation fits, and 1,005 patient-nested inner fits, in addition to pilot and clinical searches.

## Execution

```bash
python run_all.py --mode audit
python run_all.py --mode resume
python run_all.py --mode compute
python run_all.py --mode report
```

`resume` uses file/row-count completion contracts and executes compute plus reporting. Every stage has a separate log and the run can continue after interruption.

## Leakage controls

- The official internal 1,100/195/229 split is subject-disjoint; exact IDs are committed.
- Development screening excludes the official test partition.
- Repeated grouped OOF uses three fresh seeds and five patient-wise folds.
- The primary result uses patient-nested selection (201 candidates in each outer fold); the repeated grouped OOF is secondary because prior design selection was not fully nested.
- External affine calibration uses patient-level cross-fitting and is explicitly supervised.
- VitalDB case selection is outcome-blind and locked before evaluation.

## Claim traceability

`artifacts/FINAL_COMPLETION_AUDIT.json` checks split, duplicate policy, run counts, leakage replay, tables, figures, and manuscript artifacts and exits nonzero on any mismatch.

## Decision policy

The nested procedure reports whichever configuration is independently selected inside each outer fold; it does not force a positive result. The historical 75-fit decision is preserved as a secondary result. Any result crossing a statistical boundary requires a documented new analysis rather than silent replacement.

## Paper build

The runner regenerates all tables, four figure families, `paper/main_reviewer_corrected.pdf`, an editable DOCX, and a PDF supplement. Tectonic is required for the report phase.
