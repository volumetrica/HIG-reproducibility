"""Run the generic real-image HIG pipeline from a metadata CSV."""

from argparse import ArgumentParser
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hig_real_predictive_pipeline import extract_dataset, evaluate_representations


def main():
    ap = ArgumentParser()
    ap.add_argument("metadata", help="CSV with sample_id,image_path,label and optional group/variant/segmentation_path")
    ap.add_argument("--outdir", default="outputs/generic")
    ap.add_argument("--segmentation-mode", choices=["uniform", "multiotsu"], default="multiotsu")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata)

    features = extract_dataset(
        meta,
        segmentation_mode=args.segmentation_mode,
        levels=args.levels,
        validate_euler=True,
    )
    features.to_csv(out / "hig_features.csv", index=False)

    summary, repeat_df, pred_df, mapping = evaluate_representations(
        features,
        group_col="group" if "group" in features.columns else None,
        k=args.folds,
        repeats=args.repeats,
        inner_k=args.inner_folds,
        seed=args.seed,
    )
    summary.to_csv(out / "predictive_performance.csv", index=False)
    repeat_df.to_csv(out / "repeat_metrics.csv", index=False)
    pred_df.to_csv(out / "oof_predictions.csv", index=False)
    print("Label mapping:", mapping)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
