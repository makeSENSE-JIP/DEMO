"""Compare the EnIF-MDA prior-precision estimators on the 5SPOT benchmark.

Reads two run directories whose EnIF-MDA stages used different
--precision-estimator settings (approximate and complete), each carrying the
same low-rank GN record, and writes combined metrics, the misfit-vs-runtime
data and the comparison figure used by README.md.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from opm_ert_demo.collect import (
    collect_enif,
    collect_lowrank,
    enif_trajectory,
    gn_trajectory,
    metric_rows,
    write_metrics,
)
from opm_ert_demo.observations import Observations
from opm_ert_demo.objective_plot import enif_points, gn_points

VARIANTS = (
    ("approximate", "tab:blue", "s-"),
    ("complete", "tab:green", "D-"),
)


def _plot_prediction(ax, obs, key, posteriors):
    columns = obs.columns(key)
    key_index = obs.keys.index(key)
    for label, predictions, color, style in posteriors:
        values = predictions[:, columns]
        ax.plot(obs.dates, np.median(values, axis=0), style, color=color, label=label)
        ax.fill_between(obs.dates, np.quantile(values, 0.1, axis=0),
                        np.quantile(values, 0.9, axis=0), color=color, alpha=0.15)
    ax.errorbar(obs.dates, obs.values[:, key_index], yerr=obs.stds[:, key_index],
                fmt="k.", label="observations", capsize=3)
    ax.set_title(f"{key} posterior predictions")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _plot_field_triple(ax, fig, panels, panel_title):
    ax.axis("off")
    vmin = min(float(field.min()) for field, _ in panels)
    vmax = max(float(field.max()) for field, _ in panels)
    for index, (field, title) in enumerate(panels):
        subplot = ax.inset_axes([0.03 + 0.33 * index, 0.05, 0.29, 0.80])
        image = subplot.imshow(field, origin="lower", cmap="viridis", vmin=vmin,
                               vmax=vmax, aspect="equal")
        subplot.set_title(title, fontsize=9)
        subplot.set_xticks([])
        subplot.set_yticks([])
        fig.colorbar(image, ax=subplot, fraction=0.04, pad=0.02)
    ax.set_title(panel_title, fontsize=11)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approximate-run-dir", type=Path, required=True)
    parser.add_argument("--complete-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for label, run_dir in (("approximate", args.approximate_run_dir),
                           ("complete", args.complete_run_dir)):
        run_dir = run_dir.resolve()
        manifest = json.loads((run_dir / "run.json").read_text())
        record = manifest["methods"]["enif"]
        if record.get("precision_estimator") != label:
            raise SystemExit(
                f"{run_dir} was run with estimator {record.get('precision_estimator')!r},"
                f" expected {label!r}")
        obs = Observations.load(run_dir / "reference/observations.json")
        runs[label] = {
            "dir": run_dir,
            "manifest": manifest,
            "obs": obs,
            "enif": collect_enif(run_dir, obs, manifest),
            "trajectory": enif_trajectory(run_dir, manifest, obs),
            "points": enif_points(run_dir, manifest, obs),
        }

    run_a = runs["approximate"]["dir"]
    manifest_a = runs["approximate"]["manifest"]
    obs = runs["approximate"]["obs"]
    lowrank = collect_lowrank(run_a, obs, manifest_a)

    args.output.mkdir(parents=True, exist_ok=True)

    series = {"LowRank-GN": gn_points(run_a, manifest_a, obs)}
    series.update(
        {f"EnIF-MDA ({label})": runs[label]["points"] for label, _, _ in VARIANTS})
    with (args.output / "objective_vs_runtime.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "phase", "iteration",
                         "elapsed_seconds", "data_misfit_objective"])
        for name, points in series.items():
            for seconds, misfit, phase, iteration in points:
                writer.writerow([name, phase, iteration, seconds, misfit])

    ensembles = {
        "EnIF prior": runs["approximate"]["enif"]["prior"],
        "EnIF-MDA (approximate) posterior": runs["approximate"]["enif"]["posterior"],
        "EnIF-MDA (complete) posterior": runs["complete"]["enif"]["posterior"],
        "LowRank-GN posterior": lowrank["posterior"],
    }
    write_metrics(args.output / "metrics.csv",
                  metric_rows(obs, ensembles, runs["approximate"]["enif"]["prior"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    figure, axis = plt.subplots(figsize=(9, 6))
    for name, points in series.items():
        axis.plot([p[0] for p in points], [p[1] for p in points], "o-", label=name)
        for x, y, phase, iteration in points:
            if phase == "mda_update":
                axis.annotate(f"MDA {iteration}", (x, y), textcoords="offset points",
                              xytext=(0, 7), fontsize=7)
    axis.set_yscale("log")
    axis.set_xlabel("Elapsed serial wall-clock time (s)")
    axis.set_ylabel("Objective data misfit")
    axis.set_title("Misfit vs runtime: EnIF-MDA precision estimators")
    axis.grid(alpha=0.3, linestyle="--", which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "objective_vs_runtime.png", dpi=160)
    print(f"figure: {args.output / 'objective_vs_runtime.png'}")

    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    gn_traj = gn_trajectory(run_a, manifest_a)
    ax.plot([p[0] for p in gn_traj], [p[1] for p in gn_traj], "o-",
            color="tab:orange", label="LowRank-GN", markersize=4)
    for label, color, style in VARIANTS:
        points = runs[label]["trajectory"]
        ax.plot([p[0] for p in points], [p[1] for p in points], style,
                color=color, label=f"EnIF-MDA ({label})", markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Elapsed serial wall-clock time (s)")
    ax.set_ylabel("Data misfit")
    ax.set_title("Misfit vs runtime")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--", which="both")

    posteriors = [(f"EnIF-MDA ({label})", runs[label]["enif"]["posterior"], color, style)
                  for label, color, style in VARIANTS]
    posteriors.append(("LowRank-GN posterior", lowrank["posterior"], "tab:orange", "-"))

    ax = fig.add_subplot(gs[0, 1])
    _plot_prediction(ax, obs, "WWPR:P2", posteriors)

    record = manifest_a["methods"]["lowrank_gn"]
    laplace = np.load(run_a / record["artifacts"]["laplace"])
    gn_fields = np.array([s.reshape(50, 50, order="F") for s in laplace["samples"].T])
    approx_fields = runs["approximate"]["enif"]["posterior_fields"]
    complete_fields = runs["complete"]["enif"]["posterior_fields"]

    ax = fig.add_subplot(gs[1, 0])
    _plot_field_triple(ax, fig, [
        (approx_fields.std(axis=0), "EnIF approx std"),
        (complete_fields.std(axis=0), "EnIF complete std"),
        (gn_fields.std(axis=0), "LowRank-GN std"),
    ], "Posterior standard deviation (log-PERMX field)")

    ax = fig.add_subplot(gs[1, 1])
    _plot_prediction(ax, obs, "WOPR:P1", posteriors)

    ax = fig.add_subplot(gs[2, 0])
    _plot_field_triple(ax, fig, [
        (approx_fields.mean(axis=0), "EnIF approx mean"),
        (complete_fields.mean(axis=0), "EnIF complete mean"),
        (gn_fields.mean(axis=0), "LowRank-GN mean"),
    ], "Posterior mean (log-PERMX field)")

    ax = fig.add_subplot(gs[2, 1])
    _plot_prediction(ax, obs, "WWPR:P1", posteriors)

    fig.suptitle("5SPOT benchmark: EnIF-MDA precision estimators vs low-rank Gauss-Newton",
                 fontsize=13)
    fig.savefig(args.output / "comparison.png", dpi=160, bbox_inches="tight")
    print(f"figure: {args.output / 'comparison.png'}")

    for label, _, _ in VARIANTS:
        timing = runs[label]["manifest"]["methods"]["enif"]["timing"]
        print(f"EnIF-MDA ({label}): total={timing['total_seconds']:.0f}s "
              f"prior={timing['prior_evaluation_seconds']:.0f}s "
              f"updates={timing['update_seconds']:.1f}s "
              f"evaluations={timing['evaluation_seconds']:.0f}s")


if __name__ == "__main__":
    main()
