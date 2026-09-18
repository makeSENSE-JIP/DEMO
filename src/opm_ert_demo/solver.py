"""Fixed prior-output-basis GN policy from the parent serial benchmark."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ert.analysis.low_rank_gn import prediction_basis, woodbury_step


@dataclass
class Fit:
    evaluation: object
    basis: np.ndarray
    jacobian: np.ndarray
    history: list[dict]
    converged: bool
    stop_reason: str


def fit_fixed_gn(prior, observations, errors, evaluate, batch_vjp, settings, rng, checkpoint):
    """Evaluate accepts (parameters x members, record); batch_vjp takes a list of weights."""
    started = time.perf_counter()
    x = np.clip(prior.mean + prior.sample(1, rng)[:, 0], *settings.log_bounds)
    candidates = np.clip(prior.mean[:, None] + prior.sample(settings.basis_members, rng), *settings.log_bounds)
    predictions = np.column_stack([e.predictions for e in evaluate(candidates, False)])
    basis = prediction_basis(predictions, errors, energy=settings.energy, rank=settings.rank)
    center = evaluate(x[:, None], True)[0]
    recorded = True
    damping = settings.damping
    history = []
    converged = False
    stop_reason = "iteration_limit"

    def objectives(evaluation):
        residual = (evaluation.predictions - observations) / errors
        delta = evaluation.parameters - prior.mean
        data = float(residual @ residual / 2)
        return data + float(delta @ prior.apply(delta) / 2), data

    def derivatives(evaluation):
        grad_weight = (evaluation.predictions - observations) / errors**2
        mode_weights = [col / errors for col in basis.T] if basis.shape[1] else []
        results = batch_vjp(evaluation, [grad_weight, *mode_weights])
        gradient = results[0] + prior.apply(evaluation.parameters - prior.mean)
        jacobian = np.stack(results[1:]) if basis.shape[1] else np.empty((0, prior.mean.size))
        return gradient, jacobian

    for iteration in range(settings.max_iterations):
        if not recorded:
            center = evaluate(center.parameters[:, None], True)[0]
            recorded = True
        gradient, jacobian = derivatives(center)
        norm = float(np.linalg.norm(gradient))
        before, data_before = objectives(center)
        accepted = False
        step_norm = 0.0
        if norm < settings.gradient_tolerance:
            converged, stop_reason = True, "gradient_tolerance"
        else:
            step = woodbury_step(prior, jacobian, gradient, damping)
            for alpha in settings.alphas:
                candidate = np.clip(center.parameters + alpha * step, *settings.log_bounds)
                if np.array_equal(candidate, center.parameters):
                    continue
                try:
                    trial = evaluate(candidate[:, None], False)[0]
                except Exception:
                    print(f"line-search trial alpha={alpha} failed; rejecting step", flush=True)
                    continue
                if objectives(trial)[0] < before:
                    step_norm = float(np.linalg.norm(candidate - center.parameters))
                    center, recorded, accepted = trial, False, True
                    break
            if accepted and step_norm < settings.step_tolerance:
                converged, stop_reason = True, "step_tolerance"
        objective, data_misfit = objectives(center)
        row = dict(iteration=iteration, objective_before=before, objective=objective,
                   data_misfit_before=data_before, data_misfit=data_misfit,
                   gradient_norm=norm, step_norm=step_norm, damping=damping,
                   rank=basis.shape[1], accepted=accepted,
                   elapsed_seconds=time.perf_counter() - started)
        history.append(row)
        checkpoint(center, row)
        if converged:
            break
        damping = max(damping / 10, 1e-6) if accepted else min(damping * 10, 1e12)
        if damping >= 1e12:
            stop_reason = "damping_limit"
            break
    if not recorded:
        center = evaluate(center.parameters[:, None], True)[0]
    gradient, jacobian = derivatives(center)
    if float(np.linalg.norm(gradient)) < settings.gradient_tolerance:
        converged, stop_reason = True, "gradient_tolerance"
    return Fit(center, basis, jacobian, history, converged, stop_reason)


def posterior_samples(prior, fit, draws, eigenvalue_tolerance=1e-5):
    """Undamped Laplace samples using the benchmark's relative eigenvalue cutoff."""
    inverse = prior.solve(fit.jacobian.T)
    gram = fit.jacobian @ inverse
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    threshold = eigenvalue_tolerance * max(float(values[-1]) if len(values) else 0, 1.0)
    keep = np.flatnonzero(values > threshold)[::-1]
    values = values[keep]
    vectors = vectors[:, keep] / np.sqrt(values)[None, :]
    basis = inverse @ vectors
    precision_basis = fit.jacobian.T @ vectors
    samples = fit.evaluation.parameters[:, None] + draws + basis @ (
        (1 / np.sqrt(1 + values) - 1)[:, None] * (precision_basis.T @ draws)
    )
    return samples, values, basis
