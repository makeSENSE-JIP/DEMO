# opm-adjoint-chainrule — provisional ownership, explicit mathematics

This separately installable package contains the permeability chain rule required
by the 5SPOT demo. Its eventual home (ERT or OPM) is **not decided**. It imports
neither project, installs no hooks, and does not patch either project's sources.

## Interface

`PermeabilityMap(parameters, matrix)` describes `K_mD = A @ exp(m)` per active
cell. Matrix rows are **PERMX, PERMY, PERMZ**; columns follow `parameters`.
`physical_permeability(m)` returns the three physical components in mD.
`pullback(m, gradient_si)` consumes OPM's three partial derivatives with respect
to physical permeability **in m²**, and returns derivatives with respect to the
independent **natural-log mD parameters**. Arrays are cell-major, with no cell
reordering. Invalid shapes, nonfinite values, or degenerate maps raise ValueError.

For the demo, `A = [[1], [1], [0.001]]`, so

```
dJ/dlog(PERMX) = exp(m) * 9.869233e-16 * (g_x + g_y + 0.001*g_z)
```

The caller must apply this exactly once, **instead of** ERT's independent-field
pullback. If OPM later emits derivatives of tied parameters directly, this
physical-component adapter must be removed or replaced to avoid double counting.
Raw OPM gradient files remain unchanged. The demo records the mapping and package
version in each run manifest and uses the same mapping to render its deck ties.

Only constant, linear physical-permeability ties with exponential independent
parameters are supported. This is not an Eclipse deck parser or an automatic
differentiation engine for arbitrary COPY/MULTIPLY/region operations.

From the DEMO directory:

```bash
python -m pip install ./packages/opm-adjoint-chainrule
python -m unittest discover -s packages/opm-adjoint-chainrule/tests -v
```
