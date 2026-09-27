# Experimental-results map

This file maps the reproducibility artifact to the experimental section of the
HIG manuscript.

| Manuscript experiment | Reproduction entry point | Reference output |
|---|---|---|
| Interactive intrinsic HIG sanity check | notebooks/00_HIG_10x10_demo.ipynb | notebook output |
| Structured rectangle generator | notebooks/01_rectangle_synthetic_generator.ipynb | generated CSV/ZIP |
| 1,300-image synthetic publication benchmark | notebooks/02_publication_synthetic_experiments.ipynb | results/synthetic/*.csv |
| BreaKHis 40X predictive experiment | notebooks/03_BreaKHis_40x_experiment.ipynb | results/breakhis_40x/*.csv |
| Generic arbitrary-size real images | notebooks/04_generic_real_image_experiment.ipynb or scripts/run_generic.py | user-selected output directory |

The real-image classification protocol compares:

1. segmentation-size baseline;
2. RAG;
3. regional contact multigraph;
4. HIG without cuts;
5. binary-incidence HIG;
6. full multiplicity-aware HIG.

The cellular quantity chi_cell is used as a construction check and is never used
as a predictive feature.
