"""Objective-vs-runtime figure for one run directory.

Follows the parent fair benchmark's convention: the plotted quantity is the
data misfit 0.5*(g-d)^T Cd^-1 (g-d) against serial elapsed seconds. Low-rank
GN contributes its MAP-iterate trajectory (all basis, record, line-search and
adjoint-replay costs are inside the measured clock, which also covers the
posterior predictive sweep at constant misfit); EnIF-MDA contributes the
ensemble-mean prediction misfit after the prior evaluation and after each
update/evaluate pass. EnIF is included automatically once its stage has run
in the same run directory.
"""

import argparse
import csv
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from .collect import ensemble_predictions
from .observations import Observations


def gn_points(run_dir: Path, manifest: dict, obs: Observations):
    record = manifest["methods"].get("lowrank_gn")
    if record is None or record["status"] != "complete":
        return None
    history = [json.loads(line) for line in
               (run_dir / record["artifacts"]["history"]).read_text().splitlines() if line.strip()]
    points = [(0.0, history[0]["data_misfit_before"], "map_start", -1)]
    points.extend((row["elapsed_seconds"], row["data_misfit"], "map_optimization", row["iteration"])
                  for row in history)
    points.append((record["timing"]["total_seconds"], history[-1]["data_misfit"],
                   "posterior_predictive", -1))
    return points


def enif_points(run_dir: Path, manifest: dict, obs: Observations):
    record = manifest["methods"].get("enif")
    if record is None or record["status"] != "complete":
        return None
    from ert.storage import open_storage

    points = []
    clock = 0.0
    with open_storage(run_dir / "enif" / "storage", mode="r") as storage:
        clock += record["timing"]["prior_evaluation_seconds"]
        prior = storage.get_ensemble(UUID(record["prior_ensemble"]))
        misfit = float(obs.misfit(ensemble_predictions(prior, obs).mean(axis=0)[None, :])[0])
        points.append((clock, misfit, "prior_evaluation", 0))
        for step in record["steps"]:
            clock += step["update_seconds"] + step["evaluate_seconds"]
            ensemble = storage.get_ensemble(UUID(step["ensemble"]))
            misfit = float(obs.misfit(
                ensemble_predictions(ensemble, obs).mean(axis=0)[None, :])[0])
            points.append((clock, misfit, "mda_update", step["iteration"]))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-history", type=Path,
                        help="optional PET+JutulDarcy misfit_history.csv to overlay")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run.json").read_text())
    obs = Observations.load(run_dir / "reference/observations.json")

    series = {}
    gn = gn_points(run_dir, manifest, obs)
    if gn:
        series["LowRank-GN"] = gn
    enif = enif_points(run_dir, manifest, obs)
    if enif:
        series["EnIF-MDA"] = enif
    if not series:
        raise SystemExit("no completed method found in the run manifest")

    if args.reference_history:
        series = {f"OPM+ERT {name}": points for name, points in series.items()}
        with args.reference_history.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for method in ("LowRank-GN", "EnIF-MDA"):
            points = [
                (float(row["elapsed_seconds"]), float(row["data_misfit_objective"]),
                 row["phase"], int(row["iteration"]))
                for row in rows if row["method"] == method
            ]
            if points:
                series[f"PET+JutulDarcy {method}"] = points

    results = run_dir / "results"
    results.mkdir(exist_ok=True)
    stem = "objective_vs_runtime_stack_comparison" if args.reference_history else "objective_vs_runtime"
    with (results / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "phase", "iteration", "elapsed_seconds", "data_misfit_objective"])
        for name, points in series.items():
            for seconds, misfit, phase, iteration in points:
                writer.writerow([name, phase, iteration, seconds, misfit])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for name, points in series.items():
        seconds = [p[0] for p in points]
        misfit = [p[1] for p in points]
        axis.plot(seconds, misfit, "o-", label=name)
        for x, y, phase, iteration in points:
            if phase == "mda_update":
                axis.annotate(f"MDA {iteration}", (x, y), textcoords="offset points",
                              xytext=(0, 7), fontsize=8)
    axis.set_yscale("log")
    axis.set_xlabel("Elapsed serial wall-clock time (s)")
    axis.set_ylabel("Objective data misfit")
    axis.set_title("Misfit vs runtime")
    axis.grid(alpha=0.3, linestyle="--", which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(results / f"{stem}.png", dpi=160)
    print(f"figure: {results / f'{stem}.png'}")
    for name, points in series.items():
        first, last = points[0], points[-1]
        print(f"{name}: {first[1]:.3f} @ {first[0]:.0f}s -> {last[1]:.3f} @ {last[0]:.0f}s")
    if enif is None:
        print("EnIF-MDA not present in this run; rerun its stage and this figure to add it.")


if __name__ == "__main__":
    main()
