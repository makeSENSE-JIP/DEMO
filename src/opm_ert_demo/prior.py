"""Reference prior plus the chain-rule adapter used by the GN driver."""

import json
from types import SimpleNamespace

import numpy as np
from opm_adjoint_chainrule import PermeabilityMap
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.sparse import load_npz


class ReferencePrior:
    """N(mu, Q^-1) with Q exported from the original benchmark (case/reference)."""

    def __init__(self, reference_dir):
        metadata = json.loads((reference_dir / "provenance.json").read_text())
        self.shape = tuple(metadata["shape"])
        self.precision = load_npz(reference_dir / "prior_precision.npz").tocsc()
        n = int(np.prod(self.shape))
        if self.precision.shape != (n, n):
            raise ValueError("Reference precision and grid dimensions disagree")
        if not np.isfinite(self.precision.data).all():
            raise ValueError("Reference precision contains non-finite values")
        asymmetry = abs(self.precision - self.precision.T)
        if asymmetry.nnz and asymmetry.max() > 1e-10 * abs(self.precision).max():
            raise ValueError("Reference precision must be symmetric")
        self.mean = np.full(n, metadata["mean_log_permx"])
        self.factor = cholesky(self.precision.toarray(), lower=True)

    def apply(self, value):
        return self.precision @ value

    def solve(self, value):
        return cho_solve((self.factor, True), value)

    def sample(self, count, rng):
        normals = rng.standard_normal((count, self.mean.size)).T
        return solve_triangular(self.factor.T, normals, lower=False)


class ChainRulePrior(ReferencePrior):
    """ReferencePrior whose permeability pullback goes through opm-adjoint-chainrule.

    ERT's OPMAdjoint delegates the physical-to-parameter gradient conversion to
    ``prior.pullback``; this adapter therefore replaces ERT's independent-field
    rule with the tied-permeability chain rule from the standalone package.
    """

    def __init__(self, reference_dir, mapping):
        super().__init__(reference_dir)
        if not isinstance(mapping, PermeabilityMap) or mapping.parameters != ("PERMX",):
            raise ValueError("the 5SPOT demo defines one independent PERMX parameter")
        self.mapping = mapping
        self.fields = [SimpleNamespace(name="PERMX")]

    def pullback(self, parameters, pv, perm):
        if pv is not None:
            raise ValueError("unexpected PV gradient: the 5SPOT deck defines no porosity ties")
        return self.mapping.pullback(np.asarray(parameters)[:, None], perm)[:, 0]
