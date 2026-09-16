# 5SPOT benchmark demo: OPM adjoints + ERT — low-rank GN vs EnIF-MDA

A self-contained reproduction of two methods from the makeSENSE 5SPOT
history-matching benchmark, driven by **OPM Flow with adjoint gradients** and
the **ERT** ensemble tool:

1. **Low-rank Gauss-Newton** — adjoint-based MAP optimization with a fixed
   prior-centered output basis, exact residual-gradient VJPs, and undamped
   low-rank Laplace posterior sampling around the MAP.
2. **EnIF-MDA** — five assimilation passes of the ensemble information filter
   with inflation weights `[5, 5, 5, 5, 5]` (matching the benchmark's
   `EnIF-MDA` configuration).

The synthetic case is the 5SPOT five-spot problem: a 50 x 50 x 1 grid, one
water injector in the centre, four BHP-controlled oil producers in the corners,
and 60 well-rate observations (WOPR/WGPR/WWPR for P1-P4 at five assimilation
dates). The uncertain parameter is natural-log PERMX. **The prior and the
observations are the original benchmark's**: a calibrated anisotropic Matern
SPDE precision (sigma = 0.5, practical range 2500 m, anisotropy 2, rotation
45 deg, mean log-PERMX = log 500) and the benchmark's truth data with its
noise rule, exported once into `case/reference/` (see Provenance below).

## Reproduce from scratch

Requirements: Linux, Python 3.12+, internet access, ~15 GB disk for the OPM
build, `bash`, `curl`, `git`, and a C++20 toolchain (provided by the install
script via micromamba).

```bash
./install_opm.sh   # 1. build OPM Flow + flow_adjoint (stamped, re-runnable)
./run_demo.sh      # 2+3. python env, both cases, collection
```

Every invocation writes a fresh, self-contained run directory
`case/runs/<RUN_ID>` (timestamped by default; override with `RUN_ID=`).
Selected stages can be rerun: `RUN_ID=<id> ./run_demo.sh collect`. Outputs:

- `results/metrics.csv` — mean-prediction misfit, predictive NLL, coverage and
  interval-width ratios, overall and per rate type (same quantities as the
  parent benchmark's post-processor)
- `results/comparison.png` — misfit distributions, GN convergence, parameter
  fields, WOPR:P1 posterior predictions
- `lowrank_gn/` — `history.jsonl`, `laplace.npz` (MAP, eigensystem, samples),
  `predict.npz` (posterior predictions, fingerprinted)
- `enif/storage/` — ERT storage with the prior and all five update iterations
- `run.json` — manifest: settings, input/executable hashes, package versions,
  chain-rule record, per-method ensemble IDs and artifacts

## The permeability chain rule (deliberately separate)

The deck ties the permeability components: `PERMY = PERMX`, `PERMZ = 0.001
PERMX`. OPM's adjoint output therefore contains three physical derivatives per
cell, and the derivative with respect to the independent log-PERMX parameter is

```
dJ/dlog(PERMX) = PERMX[mD] * 9.869233e-16 * (g_x + g_y + 0.001*g_z)
```

with `g_x/g_y/g_z` OPM's per-m^2 components. ERT's built-in pullback uses only
the PERMX column and ignores the tied components, so the demo **does not use
ERT's conversion**. Where this conversion lives (ERT or OPM) is an open design
question, so it is isolated in its own installed package:

- `packages/opm-adjoint-chainrule/` — `PermeabilityMap(parameters, matrix)`,
  pure NumPy, imports neither ERT nor OPM, with finite-difference unit tests
- `case/permeability.json` — the demo's mapping
  `{"parameters": ["PERMX"], "matrix": [[1], [1], [0.001]]}`; the same file
  renders the deck's COPY/MULTIPLY ties and drives the pullback
- The GN driver passes a prior adapter whose `pullback` ERT's `OPMAdjoint`
  calls, so the replacement happens at exactly one seam; the raw OPM gradient
  files are untouched
- Each run manifest records the package name/version and mapping

If OPM later emits tied-parameter derivatives directly, replace the adapter in
`src/opm_ert_demo/prior.py` and delete the package — nothing else changes.

## Provenance of the prior and observations

`case/reference/` holds the benchmark's calibrated prior precision
(`prior_precision.npz`), its observations (`observations.json`) and
`provenance.json` with sha256 checksums of source files. The demo validates
these checksums before every run. To regenerate them (maintainers, in the
original benchmark environment with Julia/JutulDarcy available):

```bash
code/venv_laplace/bin/python tools/export_reference.py \
  --benchmark-root <path/to/Laplace> --output case/reference
```

## Layout

| Path | Purpose |
|---|---|
| `benchmark.toml` | All case settings: seed, members, jobs, MDA weights, GN policy |
| `pyproject.toml` | Python dependencies (pinned ERT fork, matern-egrid) |
| `packages/opm-adjoint-chainrule/` | Separate chain-rule package (own pyproject + tests) |
| `install_opm.sh` | Staged build of the pinned OPM adjoint stack into `opm/` |
| `run_demo.sh` | Reproduction entry point (prepare / gn / enif / collect) |
| `case/TRUE_MODEL/`, `case/MODEL.template`, `case/permeability.json`, `case/reference/` | Immutable case inputs copied into each run |
| `src/opm_ert_demo/` | Case preparation, GN and EnIF-MDA drivers, collection |
| `tools/export_reference.py` | Maintainer export of the reference prior/observations |
| `BUILDING_ADJOINT_FLOW.md` | Background on the OPM adjoint build |

## Method notes

- **Low-rank GN** starts at a clipped prior draw, builds one output basis from
  ten prior-mean-centered candidates (energy 1.0, rank 40), takes exact
  residual-gradient VJPs plus projected curvature, Levenberg-Marquardt damping
  from 10 (x10 reject / /10 accept), a four-step halving line search, and stops
  on gradient norm 1e-3 or step norm 1e-6 — the parent benchmark's policy.
  Posterior draws use a separate stream seeded `seed + 2`, as in the parent.
- **EnIF-MDA** uses ERT's own run models in-process (the fork's CLI has no
  EnIF-MDA mode): `ensemble_experiment` loads the prior GRDECLs and evaluates,
  then each weight runs `analysis_EnIF` with `global_std_scaling = weight`
  (the ES-MDA inflation convention) followed by an ensemble evaluation.
- Recorded forwards and adjoint replays run serially (one per host); plain
  forward batches may use `jobs` workers (default 1, matching the parent's
  serial protocol).

## Tests

```bash
.venv/bin/python -m unittest discover -s packages/opm-adjoint-chainrule/tests -v
```
