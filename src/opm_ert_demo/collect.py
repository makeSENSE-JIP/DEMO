"""Collect, validate and compare both method runs inside one run directory.

Every artifact is located through the run manifest (run.json), never by
directory listing, and the GN prediction cache is validated against the
posterior samples and observations it was produced from.
"""

import argparse
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from ._common import summary_at
from .gn_case import digest
from .observations import Observations

RATE_TYPES = ("WOPR", "WGPR", "WWPR")


def ensemble_predictions(ensemble, obs: Observations) -> np.ndarray:
    stored = ensemble.experiment.observations["summary"]
    stored_order = list(zip(stored["response_key"], np.asarray(stored["time"]).astype("datetime64[D]").tolist()))
    mine = [(key, date) for date in obs.dates.tolist() for key in obs.keys]
    if stored_order != mine:
        raise RuntimeError("observation ordering mismatch against ERT storage")
    members = ensemble.ensemble_size
    series = {key: np.empty((members, len(obs.dates))) for key in obs.keys}
    for iens in range(members):
        rows = {}
        for row in ensemble.load_responses("summary", (iens,)).iter_rows(named=True):
            rows.setdefault(row["response_key"], []).append((row["time"], row["values"]))
        for key in obs.keys:
            entries = rows.get(key)
            if not entries:
                raise KeyError(f"realization {iens} has no response {key}")
            times = np.array([entry[0] for entry in entries]).astype("datetime64[D]")
            data = np.array([entry[1] for entry in entries], dtype=np.float64)
            series[key][iens] = summary_at(times, data, obs.dates)
    return np.stack([series[key] for key in obs.keys], axis=2).reshape(members, -1)


def ensemble_fields(ensemble, shape) -> np.ndarray:
    fields = []
    for iens in range(ensemble.ensemble_size):
        values = np.asarray(ensemble.load_parameters("PERMX", iens)["values"].values, dtype=np.float64)
        if values.ndim == 4:
            values = values[0]
        if values.shape != (*shape, 1):
            raise ValueError(f"stored PERMX field has shape {values.shape}, expected {(*shape, 1)}")
        fields.append(values[..., 0])
    return np.stack(fields)


def collect_enif(run_dir: Path, obs: Observations, manifest: dict) -> dict:
    from ert.storage import open_storage

    record = manifest["methods"]["enif"]
    if record["status"] != "complete":
        raise RuntimeError("EnIF run is not complete")
    with open_storage(run_dir / "enif" / "storage", mode="r") as storage:
        prior = storage.get_ensemble(UUID(record["prior_ensemble"]))
        posterior = storage.get_ensemble(UUID(record["steps"][-1]["ensemble"]))
        return {
            "members": prior.ensemble_size,
            "prior": ensemble_predictions(prior, obs),
            "posterior": ensemble_predictions(posterior, obs),
            "posterior_fields": ensemble_fields(posterior, (50, 50)),
        }


def collect_lowrank(run_dir: Path, obs: Observations, manifest: dict) -> dict:
    record = manifest["methods"]["lowrank_gn"]
    if record["status"] != "complete":
        raise RuntimeError("low-rank GN run is not complete")
    laplace = np.load(run_dir / record["artifacts"]["laplace"])
    cache = np.load(run_dir / record["artifacts"]["predictions"])
    if (cache["samples_sha256"].item() != digest(laplace["samples"])
            or cache["observations_sha256"].item() != digest(obs.values)):
        raise RuntimeError(
            "lowrank_gn/predict.npz does not match the stored posterior or observations; rerun gn_case")
    history_path = run_dir / record["artifacts"]["history"]
    return {
        "members": laplace["samples"].shape[1],
        "history": [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()],
        "map_parameters": laplace["map"],
        "map_predictions": laplace["predictions"],
        "converged": bool(laplace["converged"]),
        "stop_reason": str(laplace["stop_reason"]),
        "posterior": cache["predictions"],
    }


def true_field(case_dir: Path) -> np.ndarray:
    path = case_dir / "TRUE_MODEL" / "include" / "TRUE_MODEL.PERMX"
    tokens = []
    for line in path.read_text().splitlines():
        tokens.extend(line.split("--")[0].split())
    values = np.array([t for t in tokens if t not in ("PERMX", "/")], dtype=np.float64)
    if values.size != 50 * 50:
        raise ValueError(f"expected 2500 PERMX values, got {values.size}")
    return np.log(values.reshape(50, 50, order="F"))


def group_masks(obs: Observations) -> list[tuple[str, np.ndarray]]:
    key_mask = {rate: np.array([key.startswith(rate + ":") for key in obs.keys]) for rate in RATE_TYPES}
    masks = [("all", np.ones(obs.values.size, dtype=bool))]
    masks.extend((rate, np.tile(key_mask[rate], len(obs.dates))) for rate in RATE_TYPES)
    return masks


def predictive_nll(samples: np.ndarray, values: np.ndarray, variances: np.ndarray) -> float:
    mean = samples.mean(axis=0)
    total = np.maximum(samples.var(axis=0, ddof=1) + variances, 1e-30) if samples.shape[0] > 1 else variances
    return float(np.sum(np.log(2 * np.pi * total) / 2 + (values - mean) ** 2 / (2 * total)))


def coverage(samples: np.ndarray, values: np.ndarray, level: float) -> float:
    alpha = 1 - level
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)
    return float(np.mean((values >= low) & (values <= high)))


def width90(samples: np.ndarray) -> float:
    low, high = np.quantile(samples, [0.05, 0.95], axis=0)
    return float(np.mean(high - low))


def metric_rows(obs: Observations, ensembles: dict, prior_predictions: np.ndarray) -> list[dict]:
    flat_values = obs.values.ravel()
    flat_variances = obs.stds.ravel() ** 2
    rows = []
    for name, predictions in ensembles.items():
        for group, mask in group_masks(obs):
            subset = predictions[:, mask]
            prior_width = width90(prior_predictions[:, mask])
            row = {
                "ensemble": name, "group": group, "n_data": int(mask.sum()),
                "mean_prediction_misfit": float(
                    obs.misfit(subset.mean(axis=0)[None, :], columns=np.flatnonzero(mask))[0]),
                "predictive_nll": predictive_nll(subset, flat_values[mask], flat_variances[mask]),
                "coverage_50": coverage(subset, flat_values[mask], 0.5),
                "coverage_90": coverage(subset, flat_values[mask], 0.9),
                "coverage_95": coverage(subset, flat_values[mask], 0.95),
                "width90": width90(subset),
                "width90_over_prior": width90(subset) / prior_width if prior_width > 0 else float("nan"),
            }
            rows.append(row)
    return rows


def write_metrics(path: Path, rows: list[dict]) -> None:
    columns = ["ensemble", "group", "n_data", "mean_prediction_misfit", "predictive_nll",
               "coverage_50", "coverage_90", "coverage_95", "width90", "width90_over_prior"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(
            f"{row[column]:.6g}" if isinstance(row[column], float) else str(row[column])
            for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"metrics: {path}")
    for line in lines:
        print(" ", line)


def gn_trajectory(run_dir: Path, manifest: dict):
    record = manifest["methods"].get("lowrank_gn")
    if record is None or record["status"] != "complete":
        return None
    history = [json.loads(line) for line in
               (run_dir / record["artifacts"]["history"]).read_text().splitlines() if line.strip()]
    points = [(0.0, history[0]["data_misfit_before"])]
    points.extend((row["elapsed_seconds"], row["data_misfit"]) for row in history)
    return points


def enif_trajectory(run_dir: Path, manifest: dict, obs: Observations):
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
        points.append((clock, misfit))
        for step in record["steps"]:
            clock += step["update_seconds"] + step["evaluate_seconds"]
            ensemble = storage.get_ensemble(UUID(step["ensemble"]))
            misfit = float(obs.misfit(
                ensemble_predictions(ensemble, obs).mean(axis=0)[None, :])[0])
            points.append((clock, misfit))
    return points


def _plot_prediction(ax, obs, key, enif_post, gn_post):
    columns = obs.columns(key)
    key_index = obs.keys.index(key)
    for name, predictions, color in (
            ("EnIF posterior", enif_post, "tab:blue"),
            ("LowRank-GN posterior", gn_post, "tab:orange")):
        values = predictions[:, columns]
        ax.plot(obs.dates, np.median(values, axis=0), color=color, label=name)
        ax.fill_between(obs.dates, np.quantile(values, 0.1, axis=0),
                        np.quantile(values, 0.9, axis=0), color=color, alpha=0.2)
    ax.errorbar(obs.dates, obs.values[:, key_index], yerr=obs.stds[:, key_index],
                fmt="k.", label="observations", capsize=3)
    ax.set_title(f"{key} posterior predictions")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _plot_field_pair(ax, fig, panels, panel_title):
    ax.axis("off")
    vmin = min(float(p[0].min()) for p in panels)
    vmax = max(float(p[0].max()) for p in panels)
    for index, (field, title) in enumerate(panels):
        left = 0.05 if index == 0 else 0.55
        subplot = ax.inset_axes([left, 0.05, 0.40, 0.80])
        image = subplot.imshow(field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                               aspect="equal")
        subplot.set_title(title, fontsize=10)
        subplot.set_xticks([])
        subplot.set_yticks([])
        fig.colorbar(image, ax=subplot, fraction=0.04, pad=0.02)
    ax.set_title(panel_title, fontsize=11)


def plot_results(out: Path, run_dir: Path, obs: Observations, enif: dict, lowrank: dict,
                 provenance: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.35)

    # Row 0, col 0: data misfit vs runtime
    ax = fig.add_subplot(gs[0, 0])
    gn_traj = gn_trajectory(run_dir, json.loads((run_dir / "run.json").read_text()))
    enif_traj = enif_trajectory(run_dir, json.loads((run_dir / "run.json").read_text()), obs)
    if gn_traj:
        ax.plot([p[0] for p in gn_traj], [p[1] for p in gn_traj], "o-",
                color="tab:orange", label="LowRank-GN", markersize=4)
    if enif_traj:
        ax.plot([p[0] for p in enif_traj], [p[1] for p in enif_traj], "s-",
                color="tab:blue", label="EnIF-MDA", markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Elapsed serial wall-clock time (s)")
    ax.set_ylabel("Data misfit")
    ax.set_title("Misfit vs runtime")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--", which="both")

    # Row 0, col 1: WWPR:P2 posterior predictions (water breakthrough)
    ax = fig.add_subplot(gs[0, 1])
    _plot_prediction(ax, obs, "WWPR:P2", enif["posterior"], lowrank["posterior"])

    # Row 1, col 0: posterior standard deviation fields (log-PERMX)
    ax = fig.add_subplot(gs[1, 0])
    enif_std = enif["posterior_fields"].std(axis=0)
    record = json.loads((run_dir / "run.json").read_text())["methods"]["lowrank_gn"]
    laplace = np.load(run_dir / record["artifacts"]["laplace"])
    gn_samples = laplace["samples"]
    gn_fields = np.array([s.reshape(50, 50, order="F") for s in gn_samples.T])
    gn_field_std = gn_fields.std(axis=0)
    _plot_field_pair(ax, fig, [
        (enif_std, "EnIF posterior std"),
        (gn_field_std, "LowRank-GN posterior std"),
    ], "Posterior standard deviation (log-PERMX field)")

    # Row 1, col 1: WOPR:P1 posterior predictions
    ax = fig.add_subplot(gs[1, 1])
    _plot_prediction(ax, obs, "WOPR:P1", enif["posterior"], lowrank["posterior"])

    # Row 2, col 0: posterior mean fields (log-PERMX)
    ax = fig.add_subplot(gs[2, 0])
    enif_mean = enif["posterior_fields"].mean(axis=0)
    gn_mean = np.array([s.reshape(50, 50, order="F") for s in gn_samples.T]).mean(axis=0)
    _plot_field_pair(ax, fig, [
        (enif_mean, "EnIF posterior mean"),
        (gn_mean, "LowRank-GN posterior mean"),
    ], "Posterior mean (log-PERMX field)")

    # Row 2, col 1: WWPR:P1 posterior predictions
    ax = fig.add_subplot(gs[2, 1])
    _plot_prediction(ax, obs, "WWPR:P1", enif["posterior"], lowrank["posterior"])

    fig.suptitle("5SPOT benchmark: OPM adjoints + ERT — EnIF-MDA vs low-rank Gauss-Newton",
                 fontsize=13)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"figure: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run.json").read_text())
    obs = Observations.load(run_dir / "reference/observations.json")
    enif = collect_enif(run_dir, obs, manifest)
    lowrank = collect_lowrank(run_dir, obs, manifest)
    provenance = json.loads((run_dir / "reference/provenance.json").read_text())

    ensembles = {
        "EnIF prior": enif["prior"],
        "EnIF posterior": enif["posterior"],
        "LowRank-GN posterior": lowrank["posterior"],
    }
    results = run_dir / "results"
    results.mkdir(exist_ok=True)
    np.savez_compressed(
        results / "predictions.npz",
        enif_prior=enif["prior"], enif_posterior=enif["posterior"],
        lowrank_posterior=lowrank["posterior"], lowrank_map=lowrank["map_predictions"],
    )
    write_metrics(results / "metrics.csv", metric_rows(obs, ensembles, enif["prior"]))
    plot_results(results / "comparison.png", run_dir, obs, enif, lowrank, provenance)
    for method in ("lowrank_gn", "enif"):
        record = manifest["methods"][method]
        timing = record.get("timing", {})
        phases = {key: value for key, value in timing.items() if key.endswith("_seconds")}
        print(f"timing {method}: " + " ".join(f"{key}={value:.1f}s" for key, value in phases.items()))


if __name__ == "__main__":
    main()
