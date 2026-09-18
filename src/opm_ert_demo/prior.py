"""Reference prior plus the chain-rule adapter used by the GN driver."""

import json
from types import SimpleNamespace

import numpy as np
from opm_adjoint_chainrule import PermeabilityMap
from scipy.linalg import cho_solve, cholesky
from scipy.sparse import csc_matrix, load_npz
from scipy.sparse.linalg import spsolve_triangular


class ReferencePrior:
    """N(mu, Q^-1) with Q exported from the original benchmark (case/reference).

    ``sample`` replicates jutuldarcy's CHOLMOD prior sampler exactly: the
    factor and permutation from ``sksparse.cholmod.cho_factor(Q, lower=True)``
    are persisted in ``prior_sampler.npz`` so draws need only a sparse
    triangular solve, and every draw passes through the same Fortran-ordered
    masked-grid round trip as ``jutuldarcy.sample_prior_vectors``
    (vector -> ``(ni, nj)`` grid -> C-ordered flatten).
    """

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
        sampler_path = reference_dir / "prior_sampler.npz"
        if not sampler_path.is_file():
            raise FileNotFoundError(
                f"{sampler_path} is missing; re-prepare the run from a case "
                "reference exported with the CHOLMOD sampler factor"
            )
        with np.load(sampler_path) as sampler:
            lower = csc_matrix(
                (
                    sampler["L_data"],
                    sampler["L_indices"],
                    sampler["L_indptr"],
                ),
                shape=tuple(sampler["L_shape"]),
            )
            self._sampler_scale = sampler["d"].astype(np.float64)
            self._sampler_upper = lower.T.tocsc()
            self._sampler_perm = sampler["perm"].astype(np.intp)

    def apply(self, value):
        return self.precision @ value

    def solve(self, value):
        return cho_solve((self.factor, True), value)

    def sample(self, count, rng):
        normals = rng.standard_normal((self.mean.size, count))
        solved = spsolve_triangular(
            self._sampler_upper,
            normals / np.sqrt(self._sampler_scale)[:, None],
            lower=False,
        )
        draws = np.empty_like(solved)
        draws[self._sampler_perm, :] = solved
        grid = self.shape[:2]
        for member in range(count):
            draws[:, member] = (
                draws[:, member].reshape(grid, order="F").ravel(order="C")
            )
        return draws


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
