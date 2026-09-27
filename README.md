# HIG reproducibility artifact

Reproducibility code and reference outputs for the experimental evaluation of the
**Homological Incidence Graph (HIG)** for segmented digital images.

The repository contains:

- the intrinsic HIG implementation used in the experiments;
- an interactive 10x10 HIG demonstrator;
- the structured rectangle generator;
- the complete 1,300-image synthetic benchmark;
- the BreaKHis 40X real-image predictive experiment;
- a generic arbitrary-size real-image Python pipeline;
- compact reference tables and automated consistency checks.

## Quick start

### Google Colab

| Experiment | Colab |
|---|---|
| Interactive 10x10 HIG | [Open](https://colab.research.google.com/github/volumetrica/HIG-reproducibility/blob/main/notebooks/00_HIG_10x10_demo.ipynb) |
| Structured rectangle generator | [Open](https://colab.research.google.com/github/volumetrica/HIG-reproducibility/blob/main/notebooks/01_rectangle_synthetic_generator.ipynb) |
| Publication synthetic benchmark | [Open](https://colab.research.google.com/github/volumetrica/HIG-reproducibility/blob/main/notebooks/02_publication_synthetic_experiments.ipynb) |
| BreaKHis 40X experiment | [Open](https://colab.research.google.com/github/volumetrica/HIG-reproducibility/blob/main/notebooks/03_BreaKHis_40x_experiment.ipynb) |
| Generic real-image experiment | [Open](https://colab.research.google.com/github/volumetrica/HIG-reproducibility/blob/main/notebooks/04_generic_real_image_experiment.ipynb) |

### Local Python

```bash
git clone https://github.com/volumetrica/HIG-reproducibility.git
cd HIG-reproducibility

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/smoke_test.py
python scripts/verify_reference_results.py
```

A Conda environment is also supplied:

```bash
conda env create -f environment.yml
conda activate hig-reproducibility
```

## Repository layout

```text
HIG-reproducibility/
├── notebooks/
│   ├── 00_HIG_10x10_demo.ipynb
│   ├── 01_rectangle_synthetic_generator.ipynb
│   ├── 02_publication_synthetic_experiments.ipynb
│   ├── 03_BreaKHis_40x_experiment.ipynb
│   └── 04_generic_real_image_experiment.ipynb
├── src/
│   └── hig_real_predictive_pipeline.py
├── scripts/
│   ├── smoke_test.py
│   ├── verify_reference_results.py
│   ├── run_generic.py
│   └── execute_notebook.py
├── data/
│   ├── README.md
│   └── metadata_template.csv
├── results/
│   ├── synthetic/
│   └── breakhis_40x/
├── paper/
│   └── experimental_results.md
└── .github/workflows/smoke-test.yml
```

## 1. Controlled synthetic benchmark

The publication notebook reproduces the complete controlled benchmark with fixed
random seed `20260927`:

- `n=0`, `r=1,...,8`, 100 images per condition: **800 images**;
- `r=5`, `n=0,...,4`, 100 images per condition: **500 images**;
- total: **1,300 segmentations**.

Reference consistency result:

```text
chi_cell = |V2| - |V1| + |V0| = 2
in 1300 / 1300 synthetic segmentations
```

Compact reference tables are committed under `results/synthetic/`.

Run in Colab with notebook 02, or locally:

```bash
python scripts/execute_notebook.py synthetic
```

## 2. BreaKHis 40X experiment

The BreaKHis images are not redistributed. Download the dataset from the official
UFPR page:

https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

The paper experiment uses only the **40X** subset.

Expected parsed subset:

```text
Images:        1995
Benign:         625
Malignant:     1370
Patients:        81
Benign patients: 24
Malignant patients: 57
Magnification:  40X only
```

Fixed preprocessing:

```text
RGB
 -> hematoxylin channel
 -> Gaussian smoothing, sigma=1
 -> 5-level multi-Otsu
 -> 4-connected constant-label regions
 -> HIG/RAG descriptors
```

Prediction protocol:

```text
Outer CV:      5 patient-grouped folds x 5 repetitions
Inner CV:      4 patient-grouped folds
C grid:        1e-3, 1e-2, 1e-1, 1, 10, 100
Bootstrap:     2000 patient-level resamples
Random seed:   20260927
Classifier:    class-balanced regularized logistic regression
```

The same segmentation is used for all six representations:

1. segmentation-size baseline;
2. RAG;
3. regional contact multigraph;
4. HIG without cuts;
5. binary-incidence HIG;
6. full multiplicity-aware HIG.

The cellular quantity `chi_cell` is **not** a predictor. It is used only as a
construction-level consistency check.

Reference construction result:

```text
1995 / 1995 images processed
0 construction errors
chi_cell = 2 in all 1995 images
```

The full experiment generated 18,952,116 image-region generators and
3,527,723 cut generators.

Reference fold-wise mean ROC-AUC values are approximately:

| Representation | Mean ROC-AUC |
|---|---:|
| Segmentation-size baseline | 0.629 |
| RAG | 0.720 |
| Regional contact multigraph | 0.726 |
| HIG without cuts | 0.730 |
| Binary-incidence HIG | 0.735 |
| Full HIG | 0.736 |

The primary paired cross-fitted comparison does **not** establish a predictive
advantage of the full HIG over the RAG:

```text
Full HIG - RAG delta ROC-AUC ≈ -0.005
95% patient-bootstrap CI ≈ [-0.036, 0.028]
```

The interval contains zero. This repository therefore reproduces the reported
ablation without encoding a claim of statistically resolved superiority.

Compact reference outputs are under `results/breakhis_40x/`. Large generated
artifacts such as the per-image feature matrix and full OOF-prediction table are
recreated by the notebook rather than stored in Git.

## 3. Generic arbitrary-size images

The Python module in `src/hig_real_predictive_pipeline.py` accepts arbitrary
image dimensions and either:

- a user-provided segmentation map, or
- deterministic image quantization.

Prepare a CSV following `data/metadata_template.csv` with required columns:

```text
sample_id,image_path,label
```

and optional:

```text
group,variant,segmentation_path
```

Then run:

```bash
python scripts/run_generic.py data/metadata.csv \
    --segmentation-mode multiotsu \
    --levels 5 \
    --outdir outputs/my_experiment
```

For multiple samples from the same subject, use the `group` column so that
cross-validation keeps groups separated.

## 4. Automated verification

Every push and pull request runs a GitHub Actions smoke test that:

1. imports the intrinsic HIG implementation;
2. constructs HIGs for several synthetic segmentations;
3. verifies `chi_cell=2`;
4. checks the committed synthetic and BreaKHis summary tables against the
   expected reference values.

Run the same checks locally:

```bash
python scripts/smoke_test.py
python scripts/verify_reference_results.py
```

## Notes on interpretation

The synthetic benchmark evaluates structural behavior and implementation-level
identities. The BreaKHis experiment evaluates application-level predictive
utility under one fixed unsupervised segmentation protocol and one linear model.

These are different questions. HIG preserves typed enclosing, cut, junction,
multi-contact, and multiplicity information by construction, but the BreaKHis
ablation does not show a statistically resolved ROC-AUC improvement over the RAG
under the present protocol.

The HSF-guided realization mechanism is not part of this empirical artifact.

## BreaKHis citation

If using the BreaKHis experiment, cite the original dataset paper:

F. A. Spanhol, L. S. Oliveira, C. Petitjean, and L. Heutte,
“A Dataset for Breast Cancer Histopathological Image Classification,”
*IEEE Transactions on Biomedical Engineering*, 63(7), 1455–1462, 2016.
DOI: 10.1109/TBME.2015.2496264.

## License and citation

No software license has been added automatically. Add the license selected by the
authors before public release.

For the mapping between manuscript claims and executable artifacts, see
`paper/experimental_results.md`.
