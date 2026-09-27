"""Verify the compact reference results committed with the repository."""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

fold = pd.read_csv(ROOT / "results/breakhis_40x/predictive_fold_metrics.csv")
paired = pd.read_csv(ROOT / "results/breakhis_40x/paired_auc_differences.csv")
cls = pd.read_csv(ROOT / "results/breakhis_40x/hig_class_statistics.csv")
rect = pd.read_csv(ROOT / "results/synthetic/table_rectangle_complexity.csv")
nest = pd.read_csv(ROOT / "results/synthetic/table_nesting_complexity.csv")

assert len(fold) == 150, f"Expected 150 fold rows, got {len(fold)}"
assert set(fold["representation"]) == {
    "Segmentation-size baseline",
    "RAG",
    "Regional contact multigraph",
    "HIG without cuts",
    "Binary-incidence HIG",
    "Full HIG",
}

means = fold.groupby("representation")["roc_auc"].mean()
expected = {
    "Segmentation-size baseline": 0.629,
    "RAG": 0.720,
    "Regional contact multigraph": 0.726,
    "HIG without cuts": 0.730,
    "Binary-incidence HIG": 0.735,
    "Full HIG": 0.736,
}
for name, target in expected.items():
    assert abs(means[name] - target) < 0.01, (name, means[name], target)

row = paired.loc[paired["comparison"].str.contains("Full HIG - RAG", regex=False)]
assert len(row) == 1
d = float(row.iloc[0]["delta_auc"])
lo = float(row.iloc[0]["delta_auc_lo"])
hi = float(row.iloc[0]["delta_auc_hi"])
assert abs(d - (-0.0046)) < 0.01
assert lo < 0 < hi

assert len(rect) == 8
assert len(nest) == 5
assert {"B", "M"}.issubset(set(cls["tumor_class"]))

print("Reference-result checks passed.")
print("Mean fold ROC-AUC:")
print(means.to_string())
print(f"Full HIG - RAG paired delta AUC: {d:.4f} [{lo:.4f}, {hi:.4f}]")
