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


def plot_results(out: Path, run_dir: Path, obs: Observations, enif: dict, lowrank: dict,
                 provenance: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    enif_prior_misfit = obs.misfit(enif["prior"])
    enif_post_misfit = obs.misfit(enif["posterior"])
    gn_post_misfit = obs.misfit(lowrank["posterior"])
    gn_map_misfit = float(obs.misfit(lowrank["map_predictions"][None, :])[0])
    mean_prediction = {
        "EnIF prior": enif_prior_misfit.mean(), "EnIF posterior": enif_post_misfit.mean(),
        "LowRank-GN posterior": gn_post_misfit.mean(), "LowRank-GN MAP": gn_map_misfit,
    }

    ax = axes[0, 0]
    ax.boxplot([enif_prior_misfit, enif_post_misfit, gn_post_misfit],
               tick_labels=["EnIF prior", "EnIF posterior", "LowRank-GN post"])
    ax.scatter(range(1, 5), [mean_prediction[name] for name in
                             ("EnIF prior", "EnIF posterior", "LowRank-GN posterior", "LowRank-GN MAP")],
               marker="D", color="k", zorder=3, label="mean-prediction misfit")
    ax.set_yscale("log")
    ax.set_ylabel("objective data misfit")
    ax.set_title("Data misfit (per member; diamonds: mean prediction)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot([row["iteration"] for row in lowrank["history"]],
            [row["objective"] for row in lowrank["history"]], "o-")
    ax.set_yscale("log")
    ax.set_xlabel("GN iteration")
    ax.set_ylabel("posterior objective (data + prior)")
    ax.set_title(f"Low-rank GN (converged={lowrank['converged']}, stop={lowrank['stop_reason']})")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    truth = true_field(run_dir)
    prior_mean = np.full_like(truth, provenance["mean_log_permx"])
    enif_mean = enif["posterior_fields"].mean(axis=0)
    gn_map = lowrank["map_parameters"].reshape(50, 50, order="F")
    panels = [(truth, "True log-PERMX"), (prior_mean, "Prior mean"),
              (enif_mean, "EnIF posterior mean"), (gn_map, "LowRank-GN MAP")]
    for index, (field, title) in enumerate(panels):
        subplot = ax.inset_axes([(index % 2) * 0.5, 0.5 - (index // 2) * 0.5, 0.48, 0.45])
        center = field.mean()
        spread = max(float(field.std()) * 3, 0.1)
        image = subplot.imshow(field, origin="lower", cmap="viridis",
                               vmin=center - spread, vmax=center + spread)
        subplot.set_title(title, fontsize=9)
        subplot.set_xticks([])
        subplot.set_yticks([])
        fig.colorbar(image, ax=subplot, fraction=0.046)
    ax.axis("off")

    ax = axes[1, 1]
    columns = obs.columns("WOPR:P1")
    key_index = obs.keys.index("WOPR:P1")
    for name, predictions, color in (
            ("EnIF posterior", enif["posterior"], "tab:blue"),
            ("LowRank-GN posterior", lowrank["posterior"], "tab:orange")):
        values = predictions[:, columns]
        ax.plot(obs.dates, np.median(values, axis=0), color=color, label=name)
        ax.fill_between(obs.dates, np.quantile(values, 0.1, axis=0),
                        np.quantile(values, 0.9, axis=0), color=color, alpha=0.2)
    ax.errorbar(obs.dates, obs.values[:, key_index], yerr=obs.stds[:, key_index],
                fmt="k.", label="observations", capsize=3)
    ax.set_title("WOPR:P1 posterior predictions")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("5SPOT benchmark: OPM adjoints + ERT - EnIF-MDA vs low-rank Gauss-Newton")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
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
