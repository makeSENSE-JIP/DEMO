from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

OBS_KEYS = [
    f"{phase}:{well}"
    for phase in ("WOPR", "WGPR", "WWPR")
    for well in ("P1", "P2", "P3", "P4")
]

# Deck/report layout shared by the template and every run directory.
DECK_BASE = "MODEL"


def read_summary(base: Path, keys: list[str] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (dates64, {key: values}) from an ECLIPSE summary set."""
    from resdata.summary import Summary

    case = Summary(str(base))
    dates = case.numpy_dates
    keys = keys or OBS_KEYS
    data = {}
    for key in keys:
        if not case.has_key(key):
            raise KeyError(f"{base} summary is missing {key}")
        data[key] = np.asarray(case.get_values(key), dtype=np.float64)
    return dates, data


def summary_at(dates: np.ndarray, data: np.ndarray, wanted) -> np.ndarray:
    """Pick values at the (date) rows matching `wanted`, order preserved."""
    flat = np.asarray(dates).astype("datetime64[D]")
    data = np.asarray(data)
    if flat.ndim != 1 or data.shape != flat.shape:
        raise ValueError("Summary dates and values must be matching vectors")
    rows = []
    for date in np.asarray(wanted).astype("datetime64[D]"):
        matches = np.flatnonzero(flat == date)
        if not len(matches):
            raise KeyError(f"summary has no row for {date}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous summary: {len(matches)} rows on {date}")
        rows.append(matches[0])
    return data[rows]


def run_flow(
    executable: str,
    workdir: Path,
    deck: str = DECK_BASE,
    extra_args: tuple[str, ...] = (),
    timeout: float = 3600.0,
) -> None:
    """Run OPM Flow on `<deck>.DATA` inside `workdir`."""
    workdir = Path(workdir)
    deck_path = workdir / f"{deck}.DATA"
    if not deck_path.is_file():
        raise FileNotFoundError(deck_path)
    command = [
        executable,
        str(deck_path),
        "--output-dir=.",
        "--threads-per-process=1",
        *extra_args,
    ]
    log = workdir / "flow.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(  # noqa: S603
            command,
            cwd=workdir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
        )


def write_grdecl(path: Path, keyword: str, values: np.ndarray) -> None:
    """Write a (nx, ny, nz) array as an ECLIPSE GRDECL keyword file."""
    from ert.field_utils import FieldFileFormat, save_field

    values = np.asarray(values, dtype=np.float64)
    save_field(values, keyword, path, FieldFileFormat.GRDECL)
