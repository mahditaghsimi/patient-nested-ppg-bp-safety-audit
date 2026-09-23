# Data and redistribution status

This repository was assembled as a **local complete research snapshot**. Inclusion here records what was used; it is not proof that public redistribution is permitted.

| Resource | Local location | Role | Public-push action |
|---|---|---|---|
| MIMIC-BP-derived PPG and labels | `data/mimic_bp/raw/` | Internal training and patient-separated evaluation | Confirm the source dataset terms and institutional access conditions before redistribution. If uncertain, publish a download/preparation script and manifest instead of the arrays. |
| Patient split | `data/mimic_bp/splits/` | Reproducible subject separation | Usually safe as identifiers are pseudonymous, but confirm against the parent data agreement. |
| Forecast encoder checkpoint | `data/pretraining/` | Initialization experiment | Confirm ownership and whether source-data restrictions propagate to weights. |
| PPG-BP | `data/external/ppg_bp/` | Cuff-labelled transfer test | Preserve the original citation/license and verify redistribution permission. |
| PPG-based-BP-assessment | `data/external/ppg_ambulatory_56/` | Ambulatory cuff-labelled transfer test | Retain upstream license/citation. |
| VitalDB synchronized snapshot | `data/external/vitaldb/` | Invasive synchronized external stress test | Verify VitalDB terms and citation requirements before publishing the raw downloaded tracks. |
| PaPaGei source/checkpoint | `vendor/papagei/` and `data/pretraining/papagei_s.pt` | Foundation-model comparison | Source contains its upstream license; verify checkpoint terms and citation. |
| Referenced papers | External literature archive (not required by the runner) | Evidence archive | Do not assume PDFs may be redistributed merely because they were downloadable. Prefer DOI links in a public repository. |

Recommended public release workflow:

1. Obtain the corresponding author's and institution's approval.
2. Confirm every upstream license/terms page.
3. Remove any resource without explicit redistribution permission.
4. Keep `data/DATA_MANIFEST.json`, paths, hashes, preparation code, and citations so authorized users can reconstruct the snapshot.
5. Install Git LFS before `git add`; several files exceed normal Git hosting limits.

The models and results are for retrospective research only and must not be used for diagnosis, treatment, or patient monitoring.
