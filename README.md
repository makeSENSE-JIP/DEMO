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
  chain-rule record, per-method ensemble IDs and artifacts, and phase timings
  (GN: forward/record/replay seconds; EnIF-MDA: per-step update/evaluate
  seconds)

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

## Well-index handling: frozen COMPDAT WI (workaround for a missing adjoint term)

The 5SPOT COMPDAT entries originally defaulted the well index, so OPM computed
`WI` from the perforated cell's permeability (Peaceman: `WI ∝ sqrt(kx·ky)`).
The forward map `K → J` then contains a well-index path whose total derivative
requires, at each perforated cell,

```
dJ/dK = [dJ/dK via inter-block transmissibility]   <- what OPM's adjoint emits
      + lambda_cell · dq/dWI · dWI/dK              <- was missing (cell rows)
      + lambda_well  · dR_well/dWI · dWI/dK        <- was missing (well rows)
```

`opm-adjoint` assembles `dJ/dPERM` exclusively from the inter-cell
face-transmissibility gradient (`AdjointSolver::gatherPermGradients_`), so with
defaulted WI the extracted derivative missed the Peaceman term: measured
against record/replay finite differences, ~77–84% of the true derivative at
perforated cells was missing (~80% of a full random-direction derivative),
while interior cells and the tied-permeability composition were correct to
0.4–0.7%.

**Workaround (active).** `make-case` renders the run decks with an explicit,
tabulated COMPDAT `WI` (OPM uses deck values verbatim), freezing the well
index at the prior-mean Peaceman value computed by the formula replicated from
`opm-common`'s `WellConnections.cpp` (`src/opm_ert_demo/wellindex.py`, 718.5833
md·m for kx=ky=500 mD, 100x100x1 m cells, rw=0.25 m, skin 0). Frozen values and
inputs are recorded in each run manifest. After freezing, the same FD tests
give **−2.1% at the well cell and −0.06% at an interior cell** — the missing
term is gone; a small ε-independent residual (sub-percent to ~2%) remains at
all cells upstream and is documented below.

Caveats, recorded here for the pending upstream decision:

- OPM's *defaulted*-WI path in this build behaves as a much smaller effective
  connection factor (~5.2 md·m implied by WPI ratios and rate calibration,
  versus 718.6 from the documented formula; the rate response is non-monotone,
  so this cannot be calibrated reliably). Freezing at the formula value
  therefore changes prior-mean rates by ≲1% (the BHP producers are
  inflow-limited). The defaulted-path discrepancy is flagged for investigation
  in `opm-adjoint`/`opm-common`.
- Freezing removes the `WI(K)` dependence from the forward model, a slight
  divergence from the parent benchmark (JutulDarcy differentiates the full
  well model). The proper fix is adding the WI term to the adjoint upstream;
  then the deck can return to defaulted WI and the freeze be dropped.

## Tests

```bash
.venv/bin/python -m unittest discover -s packages/opm-adjoint-chainrule/tests -v
.venv/bin/python -m unittest discover -s src/opm_ert_demo/tests -v
```
