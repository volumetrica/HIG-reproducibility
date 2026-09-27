from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hig_real_predictive_pipeline import compute_hig_descriptors

cases = [
    np.zeros((4, 4), dtype=np.int32),
    np.array([
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,2,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ], dtype=np.int32),
    np.indices((6, 6)).sum(axis=0).astype(np.int32) % 3,
]

for i, seg in enumerate(cases):
    d = compute_hig_descriptors(seg, validate_euler=True)
    assert d["chi_cell"] == 2, (i, d["chi_cell"])

print(f"HIG smoke test passed for {len(cases)} segmentations; chi_cell=2 in all cases.")
