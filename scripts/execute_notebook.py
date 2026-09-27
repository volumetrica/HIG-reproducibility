"""Execute one repository notebook locally with nbconvert."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    "demo": ROOT / "notebooks/00_HIG_10x10_demo.ipynb",
    "rectangles": ROOT / "notebooks/01_rectangle_synthetic_generator.ipynb",
    "synthetic": ROOT / "notebooks/02_publication_synthetic_experiments.ipynb",
    "breakhis": ROOT / "notebooks/03_BreaKHis_40x_experiment.ipynb",
    "real": ROOT / "notebooks/04_generic_real_image_experiment.ipynb",
}


def main():
    ap = ArgumentParser()
    ap.add_argument("name", choices=NOTEBOOKS)
    ap.add_argument("--timeout", type=int, default=-1)
    args = ap.parse_args()
    nb = NOTEBOOKS[args.name]
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.timeout={args.timeout}",
        "--output", nb.stem + "_executed.ipynb",
        str(nb),
    ]
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
