"""Peaceman connection factors matching opm-common's Schedule/Well computation.

Replicates WellConnections.cpp for a completion along direction Z with
perpendicular permeabilities K[0]=kx, K[1]=ky (mD) and cell extents
D=(dx, dy, dz) in metres:

    rw  = diameter / 2
    r0  = 0.28 * sqrt(sqrt(ky/kx)*dx^2 + sqrt(kx/ky)*dy^2)
          / ((kx/ky)^0.25 + (ky/kx)^0.25)
    CF_SI = 2*pi * sqrt(kx*ky)[m^2] * dz / (ln(r0/min(rw, r0)) + skin)

COMPDAT's connection-transmissibility item is a Transmissibility with METRIC
unit cP*m^3/(day*bar) (= 1e-3/(86400*1e5) in SI), not md*m; the frozen value
is therefore CF_SI / (1e-3/(86400*1e5)). At the prior mean (kx=ky=500 mD,
100x100x1 m cells, diameter 0.25 m, skin 0) this is 5.2888512746, which
reproduces OPM's defaulted-WI response exactly at the prior-mean field.

A tabulated COMPDAT WI is used verbatim by OPM (CTFKind::DeckValue), so
freezing the demo's wells at this value removes the WI(K) dependence that
the adjoint gradient does not differentiate.
"""

import math

WELL_DIAMETER = 0.25
WELL_CELLS = {
    "INJ1": (25, 25),
    "P1": (1, 1),
    "P2": (1, 50),
    "P3": (50, 1),
    "P4": (50, 50),
}

MILLIDARCY_TO_M2 = 9.869233e-16
METRIC_TRANSMISSIBILITY_SI = 1e-3 / (86400.0 * 1e5)


def peaceman_well_index(kx_md, ky_md, dx_m, dy_m, dz_m,
                        diameter_m=WELL_DIAMETER, skin=0.0):
    if min(kx_md, ky_md, dx_m, dy_m, dz_m, diameter_m) <= 0:
        raise ValueError("permeabilities, extents and well diameter must be positive")
    rw_m = diameter_m / 2.0
    ratio = kx_md / ky_md
    equivalent = 0.28 * math.sqrt(
        math.sqrt(1.0 / ratio) * dx_m * dx_m + math.sqrt(ratio) * dy_m * dy_m
    ) / (ratio ** 0.25 + (1.0 / ratio) ** 0.25)
    denominator = math.log(equivalent / min(rw_m, equivalent)) + skin
    if denominator <= 0:
        raise ValueError("non-positive Peaceman denominator")
    cf_si = (2.0 * math.pi * math.sqrt(kx_md * ky_md) * MILLIDARCY_TO_M2 * dz_m
             / denominator)
    return cf_si / METRIC_TRANSMISSIBILITY_SI


def frozen_well_indexes(prior_mean_md, dx_m, dy_m, dz_m):
    if not WELL_CELLS:
        raise ValueError("no wells configured")
    return {
        name: peaceman_well_index(prior_mean_md, prior_mean_md, dx_m, dy_m, dz_m)
        for name in WELL_CELLS
    }
