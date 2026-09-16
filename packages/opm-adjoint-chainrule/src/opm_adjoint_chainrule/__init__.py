"""Explicit permeability chain rules; independent of both ERT and OPM runtimes."""

from dataclasses import dataclass

import numpy as np

MILLIDARCY_TO_M2 = 9.869233e-16


@dataclass(frozen=True)
class PermeabilityMap:
    """Linear ties K[mD] = A @ exp(m), followed by an OPM SI-gradient pullback.

    Rows of A are PERMX, PERMY, PERMZ. Columns follow ``parameters``.
    Inputs/outputs use active-cell order; no reordering or deck parsing occurs.
    """

    parameters: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        if (not self.parameters or len(set(self.parameters)) != len(self.parameters)
                or not set(self.parameters) <= {"PERMX", "PERMY", "PERMZ"}):
            raise ValueError("parameters must be unique permeability names")
        if (matrix.shape != (3, len(self.parameters))
                or not np.isfinite(matrix).all() or np.any(matrix < 0)
                or np.any(matrix.sum(axis=1) <= 0)
                or np.linalg.matrix_rank(matrix) != len(self.parameters)):
            raise ValueError("matrix must be a finite nonnegative, full-column-rank 3 x n map")

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"parameters", "matrix"}:
            raise ValueError("permeability mapping requires exactly parameters and matrix")
        return cls(tuple(data["parameters"]), tuple(tuple(row) for row in data["matrix"]))

    def to_dict(self):
        return {"parameters": list(self.parameters), "matrix": [list(row) for row in self.matrix]}

    def _physical_parameters(self, log_parameters):
        values = np.asarray(log_parameters, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.parameters):
            raise ValueError("log_parameters must have shape (active_cells, parameters)")
        with np.errstate(over="ignore", invalid="ignore"):
            physical = np.exp(values)
        if not np.isfinite(physical).all() or np.any(physical <= 0):
            raise ValueError("log_parameters must exponentiate to finite positive permeability")
        return physical

    def physical_permeability(self, log_parameters):
        """Return PERMX/Y/Z in mD, shape (active_cells, 3)."""
        result = self._physical_parameters(log_parameters) @ np.asarray(self.matrix).T
        if not np.isfinite(result).all():
            raise ValueError("mapped permeability overflowed")
        return result

    def pullback(self, log_parameters, permeability_gradient_si):
        """Return dJ/dm from OPM's three dJ/dK[m²] columns, without modifying them."""
        physical = self._physical_parameters(log_parameters)
        gradient = np.asarray(permeability_gradient_si, dtype=float)
        if gradient.shape != (physical.shape[0], 3) or not np.isfinite(gradient).all():
            raise ValueError("OPM gradient must be finite with shape (active_cells, 3)")
        result = physical * (gradient @ np.asarray(self.matrix)) * MILLIDARCY_TO_M2
        if not np.isfinite(result).all():
            raise ValueError("non-finite permeability pullback")
        return result
