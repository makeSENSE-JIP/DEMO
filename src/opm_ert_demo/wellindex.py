"""Peaceman connection factors matching opm-common's Schedule/Well computation.

Replicates WellConnections.cpp: for a completion along direction Z with
perpendicular permeabilities K[0]=kx, K[1]=ky (mD) and cell extents
D=(dx, dy, dz) in metres:

    r0  = 0.28 * sqrt(sqrt(ky/kx)*dx^2 + sqrt(kx/ky)*dy^2)
          / ((kx/ky)^0.25 + (ky/kx)^0.25)
    Kh  = sqrt(kx*ky) * dz            (md m)
    CF  = 2*pi * Kh / (ln(r0/min(rw, r0)) + skin)   (md m, METRIC COMPDAT WI)

A tabulated COMPDAT WI is used verbatim by OPM (CTFKind::DeckValue), so
freezing the demo's wells at these prior-mean values removes the WI(K)
dependence that the adjoint gradient does not differentiate.
"""

import math

WELL_RADIUS = 0.25
WELL_CELLS = {
    "INJ1": (25, 25),
    "P1": (1, 1),
    "P2": (1, 50),
    "P3": (50, 1),
    "P4": (50, 50),
}


def peaceman_well_index(kx_md, ky_md, dx_m, dy_m, dz_m, rw_m=WELL_RADIUS, skin=0.0):
    if min(kx_md, ky_md, dx_m, dy_m, dz_m, rw_m) <= 0:
        raise ValueError("permeabilities, extents and well radius must be positive")
    ratio = kx_md / ky_md
    equivalent = 0.28 * math.sqrt(
        math.sqrt(1.0 / ratio) * dx_m * dx_m + math.sqrt(ratio) * dy_m * dy_m
    ) / (ratio ** 0.25 + (1.0 / ratio) ** 0.25)
    denominator = math.log(equivalent / min(rw_m, equivalent)) + skin
    if denominator <= 0:
        raise ValueError("non-positive Peaceman denominator")
    return 2.0 * math.pi * math.sqrt(kx_md * ky_md) * dz_m / denominator


def frozen_well_indexes(prior_mean_md, dx_m, dy_m, dz_m):
    if not WELL_CELLS:
        raise ValueError("no wells configured")
    return {
        name: peaceman_well_index(prior_mean_md, prior_mean_md, dx_m, dy_m, dz_m)
        for name in WELL_CELLS
    }
