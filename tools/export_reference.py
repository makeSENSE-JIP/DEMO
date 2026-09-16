"""Maintainer-only export; run in the original benchmark's Python/Julia environment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = root / "EnIF-MDA/DATA"
    data = pd.read_pickle(data_dir / "true_data.pkl")
    variance = pd.read_pickle(data_dir / "true_data_var.pkl")
    variances = np.array([
        [entry[1] if isinstance(entry, list) else entry for entry in row]
        for row in variance.to_numpy()
    ], dtype=float)
    observations = {
        "keys": list(data.columns),
        "dates": [str(date.date()) for date in data.index],
        "values": data.to_numpy().tolist(),
        "stds": np.sqrt(variances).tolist(),
    }
    (output / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")

    import jutuldarcy as jd

    precision, mask = jd.matern_precision_from_data_file(
        str(root / "LowRank-GN-Hessian/TRUE_MODEL/TRUE_MODEL.DATA"),
        layer=1, target_variance=0.5**2, target_range=2500.0,
        target_anisotropy=2.0, target_rotation=45.0, return_mask=True,
    )
    if not np.array_equal(np.asarray(mask), np.ones((50, 50))):
        raise ValueError("Expected the all-active 50 x 50 benchmark layer")
    sp.save_npz(output / "prior_precision.npz", sp.csc_matrix(precision))
    sources = [
        data_dir / "true_data.pkl", data_dir / "true_data_var.pkl",
        root / "EnIF-MDA/PRIOR/sample_prior.py",
        root / "LowRank-GN-Hessian/run_script.py",
        root / "code/PyJutulDarcy/jutuldarcy/matern.py",
    ]
    metadata = {
        "description": "Original calibrated 2-D 5SPOT benchmark inputs; no simulator rerun",
        "ordering": "active cells in Fortran (i-fastest) order",
        "shape": [50, 50, 1], "mean_log_permx": float(np.log(500)),
        "sigma": 0.5, "range": 2500.0, "anisotropy": 2.0, "rotation_degrees": 45.0,
        "calibrate": True, "seed": 0,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "asset_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                         for name in ("observations.json", "prior_precision.npz")},
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported benchmark inputs to {output}")


if __name__ == "__main__":
    main()
