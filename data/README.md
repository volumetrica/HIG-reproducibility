# Data

The BreaKHis images are **not redistributed** in this repository.

Download BreaKHis from the official UFPR page:

https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

The real-image experiment in the paper uses the **40X** subset only.

Expected parsed subset:

- 1,995 images
- 625 benign images
- 1,370 malignant images
- 24 benign patients
- 57 malignant patients
- 81 patients total

The notebook `notebooks/03_BreaKHis_40x_experiment.ipynb` can either use the
official archive or an already-extracted 40X directory.

The generic pipeline uses `data/metadata_template.csv`.
