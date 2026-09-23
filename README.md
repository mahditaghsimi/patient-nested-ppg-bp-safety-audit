    └── vendor/papagei/       # required upstream PaPaGei code and license
```

See [`FINAL/docs/CODE_SCOPE.md`](FINAL/docs/CODE_SCOPE.md) for the distinction between the direct manuscript pipeline and preserved exploratory provenance.

## Requirements

- Python 3.11
- a CUDA-capable GPU for the full compute pipeline
- a PyTorch build compatible with the local CUDA driver
- sufficient storage for datasets, caches, checkpoints, and more than one thousand fits
- Tectonic for manuscript PDF generation
- Poppler for PDF inspection
- Git LFS for large checkpoints and arrays

Install the CUDA-compatible PyTorch build first, then install the remaining dependencies:

```bash
cd FINAL
python -m pip install -r requirements.txt
```

The pinned environment snapshot is available in [`FINAL/requirements-lock.txt`](FINAL/requirements-lock.txt).

## Data preparation

The code expects MIMIC-BP-derived PPG arrays and labels, pretrained checkpoints, and the external datasets under `FINAL/data/`. A compact layout is shown below:

```text
FINAL/data/
├── mimic_bp/
│   ├── raw/ppg/              # pXXXX_ppg.npy, shape (30, 3750)
│   ├── raw/labels/           # pXXXX_labels.npy, shape (30, 2)
│   └── splits/               # official patient-disjoint split
├── pretraining/              # forecast and PaPaGei checkpoints
└── external/
    ├── ppg_bp/
    ├── ppg_ambulatory_56/
    └── vitaldb/
```

Read [`FINAL/data/README_FA.md`](FINAL/data/README_FA.md) for exact paths and [`FINAL/DATA_LICENSES.md`](FINAL/DATA_LICENSES.md) before publishing or redistributing any data or model weights.

> Local possession of a dataset or checkpoint does not imply permission to publish it. When redistribution rights are unclear, keep the preparation code, manifests, and hashes in the public repository and omit the restricted files.

## Running the pipeline

From the code directory:

```bash
cd FINAL
python main.py
```

Useful modes:

```bash
python main.py --list-stages
python main.py --dry-run
python main.py --mode compute
python main.py --mode report
python main.py --mode audit
python main.py --from-stage innovation_search
python main.py --force vitaldb_external
```

The runner records each stage in `artifacts/pipeline_state.json`, writes separate logs to `logs/`, validates expected outputs, and can resume completed stages when their code fingerprint and output contracts remain valid.

The full run is computationally expensive: it includes the locked 200-configuration screen, 75 secondary confirmation fits, 1,005 nested inner fits, selected outer-fold fits, external evaluation, reporting, and audits. Use `--dry-run` to inspect the execution order without starting any stage.

## Outputs and traceability

| Location | Contents |
|---|---|
| [`FINAL/results/tables/`](FINAL/results/tables/) | Main and supplementary result tables |
| [`FINAL/results/figures/`](FINAL/results/figures/) | PNG and vector-PDF figures |
| [`FINAL/phase2/outputs/`](FINAL/phase2/outputs/) | Search, confirmation, nested-CV, and external-evaluation artifacts |
| [`FINAL/artifacts/`](FINAL/artifacts/) | Input, leakage, environment, state, and completion audits |
| [`FINAL/logs/`](FINAL/logs/) | Per-stage execution logs |

The pipeline preserves negative results and distinguishes the primary patient-nested estimate from the secondary repeated grouped analysis. Zero-shot evaluation and supervised target-domain adaptation are also reported separately. These distinctions are important to avoid overstating performance.

## Authors

- Mohammad Mahdi Taghsimi
- Amirhossein Arani
- Bahram Tariverdi Zadeh
- Khalil Alipour

University of Tehran, Tehran, Iran.

## Citation

If you use this repository, please cite the accompanying manuscript and the software metadata in [`FINAL/CITATION.cff`](FINAL/CITATION.cff):

```text
Taghsimi, M. M., Arani, A., Tariverdi Zadeh, B., & Alipour, K.
Patient-Nested Model Selection and Cross-Domain Safety Audit of
PPG-Only Blood-Pressure Estimation. 2026.
```

Please also cite the original MIMIC-BP, VitalDB, PPG-BP, ambulatory PPG, and PaPaGei sources when using the corresponding data or pretrained model.

## License and third-party material

The project is currently **all rights reserved** and available for private evaluation only; see [`FINAL/LICENSE`](FINAL/LICENSE). Third-party datasets, papers, code, and model weights remain subject to their original licenses and terms. PaPaGei notices are preserved in [`FINAL/THIRD_PARTY_NOTICES.md`](FINAL/THIRD_PARTY_NOTICES.md) and its vendored license.

Before making a public GitHub release, review [`FINAL/DATA_LICENSES.md`](FINAL/DATA_LICENSES.md), remove any material that cannot legally be redistributed, and use Git LFS only for files that are permitted to be published.
