"""Run the EnIF case as EnIF-MDA: five weighted update/evaluate passes.

ERT's CLI exposes one-shot EnIF only, so this driver uses ERT's own run models
in-process (ensemble_experiment to load and evaluate the prior, evaluate_ensemble
for each posterior) and the fork's analysis_EnIF with per-step global_std_scaling
equal to the ES-MDA inflation weight.
"""

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from ert.analysis._enif_update import analysis_EnIF
from ert.analysis.snapshots import SmootherSnapshot
from ert.base_model_context import use_runtime_plugins
from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE, EVALUATE_ENSEMBLE_MODE
from ert.plugins import get_site_plugins
from ert.run_models.model_factory import create_model
from ert.storage import open_storage

from .config import BenchmarkSettings
from .make_case import validate_run_inputs, write_json

PRIOR_NAME = "enif_iter_0"


def run_ert_model(config, args) -> float:
    started = time.perf_counter()
    status_queue = queue.SimpleQueue()
    with use_runtime_plugins(get_site_plugins()):
        model = create_model(config, args, status_queue)
    evaluator = EvaluatorServerConfig(port_range=None, use_ipc_protocol=True)
    thread = threading.Thread(target=model.start_simulations_thread, args=(evaluator,))
    thread.start()
    thread.join()
    failure = None
    while not status_queue.empty():
        event = status_queue.get_nowait()
        if type(event).__name__ == "EndEvent" and event.failed:
            failure = event.msg
    if failure:
        raise RuntimeError(f"ERT evaluation failed: {failure}")
    return time.perf_counter() - started


def update_step(storage_path: Path, source_id: UUID, target_name: str, iteration: int,
                weight: float, seed: int) -> tuple[UUID, float]:
    started = time.perf_counter()
    with open_storage(storage_path, mode="w") as storage:
        source = storage.get_ensemble(source_id)
        experiment = source.experiment
        target = storage.create_ensemble(
            experiment, ensemble_size=source.ensemble_size, name=target_name, iteration=iteration)
        snapshot = SmootherSnapshot(
            source_ensemble_name=source.name, target_ensemble_name=target.name,
            alpha=-1, std_cutoff=-1, global_scaling=-1)
        analysis_EnIF(
            experiment.update_parameters, experiment.observation_keys, seed, snapshot,
            source.get_realization_mask_with_responses(), source, target,
            lambda event: None, global_scaling=weight)
        return target.id, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.perf_counter()

    manifest = json.loads((run_dir / "run.json").read_text())
    validate_run_inputs(run_dir, manifest)
    settings = BenchmarkSettings.model_validate(manifest["settings"])
    flow_bin = Path(manifest["executables"]["flow"]["path"])
    os.environ["PATH"] = f"{flow_bin.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    case_dir = run_dir / "enif"
    original = Path.cwd()
    os.chdir(case_dir)
    try:
        config = ErtConfig.with_plugins(get_site_plugins()).from_file("enif.ert")
        prior_seconds = run_ert_model(config, SimpleNamespace(
            mode=ENSEMBLE_EXPERIMENT_MODE, realizations=None,
            current_ensemble=PRIOR_NAME, experiment_name="enif"))
        with open_storage(config.ens_path, mode="r") as storage:
            experiment = next(e for e in storage.experiments if e.name == "enif")
            prior_id = next(a.id for a in storage.ensembles if a.name == PRIOR_NAME
                            and a.experiment.id == experiment.id)
        steps = []
        update_seconds = 0.0
        evaluation_seconds = prior_seconds
        source = prior_id
        for index, weight in enumerate(settings.inflation):
            target_id, step_update_seconds = update_step(
                Path(config.ens_path), source, f"enif_iter_{index + 1}", index + 1,
                weight, settings.seed)
            step_evaluate_seconds = run_ert_model(config, SimpleNamespace(
                mode=EVALUATE_ENSEMBLE_MODE, realizations=None,
                ensemble_id=str(target_id)))
            update_seconds += step_update_seconds
            evaluation_seconds += step_evaluate_seconds
            steps.append({"iteration": index + 1, "weight": weight, "ensemble": str(target_id),
                          "update_seconds": step_update_seconds,
                          "evaluate_seconds": step_evaluate_seconds})
            source = target_id
    finally:
        os.chdir(original)

    total_seconds = time.perf_counter() - started
    manifest["methods"]["enif"] = {
        "status": "complete",
        "weights": settings.inflation,
        "prior_ensemble": str(prior_id),
        "steps": steps,
        "elapsed_seconds": total_seconds,
        "timing": {"total_seconds": total_seconds, "steps": len(steps),
                   "prior_evaluation_seconds": prior_seconds,
                   "update_seconds": update_seconds,
                   "evaluation_seconds": evaluation_seconds},
    }
    write_json(run_dir / "run.json", manifest)
    print(f"EnIF-MDA: {len(steps)} updates with weights {settings.inflation}")


if __name__ == "__main__":
    main()
