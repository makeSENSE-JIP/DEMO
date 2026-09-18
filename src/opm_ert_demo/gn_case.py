"""Run the low-rank GN case: fixed prior basis, adjoint VJPs, Laplace sampling.

Drives OPM flow_adjoint directly (recorded forwards, plain forwards) and uses
ERT's OPMAdjoint for replay VJPs; the physical-to-log-parameter conversion is
delegated to prior.pullback, i.e. the opm-adjoint-chainrule package.
"""

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from ert.analysis.low_rank_gn import Evaluation
from ert.analysis.opm_adjoint import OPMAdjoint
from ert.config.low_rank_gn import OPMAdjointSettings
from ert.run_arg import RunArg
from opm_adjoint_chainrule import PermeabilityMap

from ._common import read_summary, run_flow, summary_at, write_grdecl
from .config import BenchmarkSettings
from .make_case import validate_run_inputs, write_json
from .observations import Observations
from .prior import ChainRulePrior
from .solver import fit_fixed_gn, posterior_samples


def digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def observations_frame(obs: Observations) -> pl.DataFrame:
    dates = obs.dates.astype("datetime64[ns]")
    return pl.DataFrame({
        "response_key": np.repeat(np.array(obs.keys), len(obs.dates)),
        "time": np.tile(dates, len(obs.keys)),
        "observations": obs.values.T.ravel(),
        "std": obs.stds.T.ravel(),
    }).sort("time", "response_key")


class ForwardBatch:
    TIMEOUT_SECONDS = 600.0

    def __init__(self, root: Path, template: str, obs: Observations, shape, executable, arguments, jobs: int):
        self.root = root
        self.template = template
        self.obs = obs
        self.shape = shape
        self.executable = executable
        self.arguments = list(arguments)
        self.jobs = jobs
        self.batch = 0
        self.timing = {"forward_seconds": 0.0, "forward_runs": 0,
                       "record_seconds": 0.0, "record_runs": 0}

    def _member(self, index: int, column: np.ndarray, record: bool) -> Evaluation:
        workdir = self.root / f"runs/b{self.batch}-r{index}"
        workdir.mkdir(parents=True, exist_ok=False)
        (workdir / "MODEL.DATA").write_text(self.template, encoding="utf-8")
        write_grdecl(workdir / "permx.grdecl", "PERMX", np.exp(column).reshape(self.shape, order="F"))
        extra = [a for a in self.arguments if not a.startswith("--threads-per-process")]
        if record:
            extra += ["--adjoint-file=adjoint_archive", "--adjoint-save=true"]
        started = time.perf_counter()
        run_flow(self.executable, workdir, extra_args=tuple(extra), timeout=self.TIMEOUT_SECONDS)
        kind = "record" if record else "forward"
        self.timing[f"{kind}_seconds"] += time.perf_counter() - started
        self.timing[f"{kind}_runs"] += 1
        dates, data = read_summary(workdir / "MODEL")
        by_key = {key: summary_at(dates, data[key], self.obs.dates) for key in self.obs.keys}
        predictions = np.stack([by_key[key] for key in self.obs.keys], axis=1).ravel()
        context = RunArg(str(workdir), None, index, self.batch, str(workdir), "MODEL")
        return Evaluation(column.copy(), predictions, context)

    def __call__(self, parameters: np.ndarray, record: bool):
        count = parameters.shape[1]
        self.batch += 1
        if record or self.jobs == 1:
            evaluations = [self._member(i, parameters[:, i], record) for i in range(count)]
        else:
            with ThreadPoolExecutor(max_workers=self.jobs) as pool:
                evaluations = list(pool.map(lambda i: self._member(i, parameters[:, i], False), range(count)))
        return evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.perf_counter()

    manifest = json.loads((run_dir / "run.json").read_text())
    validate_run_inputs(run_dir, manifest)
    settings = BenchmarkSettings.model_validate(manifest["settings"])
    obs = Observations.load(run_dir / "reference/observations.json")
    mapping = PermeabilityMap.from_dict(manifest["chain_rule"]["mapping"])
    prior = ChainRulePrior(run_dir / "reference", mapping)

    gn_dir = run_dir / "lowrank_gn"
    adjoint_settings = OPMAdjointSettings.model_validate(
        json.loads((gn_dir / "gn.json").read_text())["adjoint"])
    adjoint = OPMAdjoint(
        adjoint_settings, prior, observations_frame(obs), threading.Event(), {}, {"<ECLBASE>": "MODEL"},
    )
    forward = ForwardBatch(
        gn_dir, (run_dir / "MODEL.template").read_text(encoding="utf-8"), obs, prior.shape,
        adjoint_settings.executable, adjoint_settings.arguments, settings.jobs,
    )

    history_path = gn_dir / "history.jsonl"
    history_path.write_text("")

    def checkpoint(_, row):
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    rng = np.random.default_rng(settings.seed)
    vjp_timing = {"vjp_seconds": 0.0, "vjp_calls": 0, "vjp_objectives": 0}

    def timed_batch_vjp(evaluation, weights):
        started = time.perf_counter()
        try:
            return adjoint.batch_vjp(evaluation, weights)
        finally:
            vjp_timing["vjp_seconds"] += time.perf_counter() - started
            vjp_timing["vjp_calls"] += 1
            vjp_timing["vjp_objectives"] += len(weights)

    fit = fit_fixed_gn(
        prior, obs.values.ravel(), obs.stds.ravel(), forward, timed_batch_vjp, settings.gn, rng, checkpoint,
    )
    sampling_rng = np.random.default_rng(settings.seed + 2)
    draws = prior.sample(settings.members, sampling_rng)
    samples, eigenvalues, sample_basis = posterior_samples(prior, fit, draws)

    posterior_predictions = np.stack([e.predictions for e in forward(samples, False)], axis=0)
    (gn_dir / "map").mkdir(exist_ok=True)
    write_grdecl(gn_dir / "map/permx.grdecl", "PERMX",
                 np.exp(fit.evaluation.parameters).reshape(prior.shape, order="F"))
    np.savez_compressed(
        gn_dir / "laplace.npz",
        map=fit.evaluation.parameters, predictions=fit.evaluation.predictions,
        samples=samples, eigenvalues=eigenvalues, sample_basis=sample_basis,
        basis=fit.basis, jacobian=fit.jacobian,
        converged=fit.converged, stop_reason=fit.stop_reason,
    )
    np.savez_compressed(
        gn_dir / "predict.npz", predictions=posterior_predictions,
        samples_sha256=digest(samples), observations_sha256=digest(obs.values),
    )

    manifest["methods"]["lowrank_gn"] = {
        "status": "complete",
        "converged": fit.converged,
        "stop_reason": fit.stop_reason,
        "iterations": len(fit.history),
        "map_data_misfit": float(obs.misfit(fit.evaluation.predictions[None, :])[0]),
        "posterior_mean_prediction_misfit": float(obs.misfit(posterior_predictions.mean(axis=0)[None, :])[0]),
        "elapsed_seconds": time.perf_counter() - started,
        "timing": {"total_seconds": time.perf_counter() - started,
                   "iterations": len(fit.history), **forward.timing, **vjp_timing},
        "artifacts": {
            "history": "lowrank_gn/history.jsonl",
            "laplace": "lowrank_gn/laplace.npz",
            "predictions": "lowrank_gn/predict.npz",
            "map_field": "lowrank_gn/map/permx.grdecl",
        },
    }
    write_json(run_dir / "run.json", manifest)
    print(f"low-rank GN: converged={fit.converged} ({fit.stop_reason}) after {len(fit.history)} iterations")


if __name__ == "__main__":
    main()
