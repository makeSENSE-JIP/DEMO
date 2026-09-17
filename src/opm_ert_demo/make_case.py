"""Prepare a fresh, self-contained run from immutable benchmark inputs."""

import argparse
import hashlib
import json
import math
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from ert.config.low_rank_gn import OPMAdjointSettings
from opm_adjoint_chainrule import PermeabilityMap

from ._common import write_grdecl
from .config import BenchmarkSettings
from .observations import Observations
from .prior import ReferencePrior
from .wellindex import frozen_well_indexes

CELL_EXTENTS_M = (100.0, 100.0, 1.0)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def input_hashes(root):
    files = [root / "MODEL.template", root / "permeability.json"]
    for directory in ("TRUE_MODEL", "reference"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file())
    return {str(path.relative_to(root)): sha256(path) for path in sorted(files)}


def _package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def validate_run_inputs(root, manifest):
    if input_hashes(root) != manifest["input_sha256"]:
        raise ValueError("Run inputs changed after preparation; create a fresh run")


def prepare_case(source, target, settings, flow, adjoint):
    if target.exists():
        raise FileExistsError(f"Run directory already exists: {target}; choose a new --run-id")
    reference = json.loads((source / "reference/provenance.json").read_text())
    for name, expected in reference["asset_sha256"].items():
        if sha256(source / "reference" / name) != expected:
            raise ValueError(f"Reference asset checksum mismatch: {name}")
    mapping = PermeabilityMap.from_dict(json.loads((source / "permeability.json").read_text()))
    if mapping.parameters != ("PERMX",) or mapping.matrix[0] != (1.0,):
        raise ValueError("5SPOT requires PERMX as the independent permeability")
    for path in (flow, adjoint):
        if not path.is_file():
            raise FileNotFoundError(f"Build OPM first; missing {path}")
    target.mkdir(parents=True)
    for name in ("TRUE_MODEL", "reference"):
        shutil.copytree(source / name, target / name)
    shutil.copy2(source / "permeability.json", target / "permeability.json")
    ties = "COPY\n PERMX PERMY /\n PERMX PERMZ /\n/\nMULTIPLY\n"
    ties += f" PERMY {mapping.matrix[1][0]:.17g} /\n PERMZ {mapping.matrix[2][0]:.17g} /\n/"
    template = (source / "MODEL.template").read_text()
    if template.count("@PERMEABILITY_TIES@") != 1:
        raise ValueError("MODEL.template requires one @PERMEABILITY_TIES@ marker")
    template = template.replace("@PERMEABILITY_TIES@", ties)
    prior_mean_md = math.exp(reference["mean_log_permx"])
    well_index = frozen_well_indexes(prior_mean_md, CELL_EXTENTS_M[0], CELL_EXTENTS_M[1], CELL_EXTENTS_M[2])
    for name in well_index:
        marker = f"@WI_{name}@"
        if template.count(marker) != 1:
            raise ValueError(f"MODEL.template requires one {marker} marker")
        template = template.replace(marker, f"{well_index[name]:.10g}")
    (target / "MODEL.template").write_text(template)
    share = target / "share"
    (share / "prior").mkdir(parents=True)
    obs = Observations.load(target / "reference/observations.json")
    obs.write_ert(share / "obs.txt")
    prior = ReferencePrior(target / "reference")
    draws = prior.sample(settings.members, np.random.default_rng(settings.seed))
    write_grdecl(share / "prior/permx_mean.grdecl", "PERMX", np.exp(prior.mean).reshape(prior.shape, order="F"))
    for member, values in enumerate((prior.mean[:, None] + draws).T):
        write_grdecl(share / f"prior/permx_{member}.grdecl", "PERMX", np.exp(values).reshape(prior.shape, order="F"))
    strict = OPMAdjointSettings().arguments
    (target / "lowrank_gn").mkdir()
    write_json(target / "lowrank_gn/gn.json", {
        "adjoint": {"executable": str(adjoint), "arguments": strict},
    })
    (target / "enif").mkdir()
    config = [f"NUM_REALIZATIONS {settings.members}", f"MIN_REALIZATIONS {settings.members}",
              f"RANDOM_SEED {settings.seed}", "QUEUE_SYSTEM LOCAL",
              f"QUEUE_OPTION LOCAL MAX_RUNNING {settings.jobs}", "NUM_CPU 1",
              "RUNPATH runs/real-<IENS>-iter-<ITER>", "GRID ../TRUE_MODEL/TRUE_MODEL.EGRID",
              "ECLBASE MODEL", "ENSPATH storage", "OBS_CONFIG ../share/obs.txt",
              *[f"SUMMARY {key}" for key in obs.keys],
              "RUN_TEMPLATE ../MODEL.template MODEL.DATA",
              "FIELD PERMX PARAMETER permx.grdecl INIT_FILES:../share/prior/permx_<IENS>.grdecl INIT_TRANSFORM:LOG OUTPUT_TRANSFORM:EXP",
              "FORWARD_MODEL FLOW"]
    (target / "enif/enif.ert").write_text("\n".join(config) + "\n")
    manifest = {
        "schema_version": 1, "status": "prepared", "settings": settings.model_dump(mode="json"),
        "chain_rule": {"package": "opm-adjoint-chainrule", "version": version("opm-adjoint-chainrule"),
                       "mapping": mapping.to_dict(), "gradient_input_units": "per m^2",
                       "parameter_units": "natural-log mD", "ownership": "provisional standalone package"},
        "prior": "reference/prior_precision.npz", "observations": "reference/observations.json",
        "frozen_well_index": {
            "reason": "adjoint dJ/dPERM omits dWI/dK; tabulated WI removes the WI(K) path",
            "formula": "Peaceman (opm-common WellConnections.cpp), METRIC cP*m3/(day*bar)",
            "permeability_md": prior_mean_md, "cell_extents_m": list(CELL_EXTENTS_M),
            "well_diameter_m": 0.25, "skin": 0.0,
            "values": well_index,
        },
        "input_sha256": input_hashes(target),
        "executables": {name: {"path": str(path), "sha256": sha256(path)}
                        for name, path in (("flow", flow), ("flow_adjoint", adjoint))},
        "packages": _package_versions(
            ("opm-ert-demo", "opm-adjoint-chainrule", "ert", "numpy", "scipy", "resdata", "graphite-maps")),
        "methods": {},
    }
    write_json(target / "run.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--adjoint", type=Path, required=True)
    args = parser.parse_args()
    prepare_case(args.source.resolve(), args.output.resolve(), BenchmarkSettings.from_file(args.config), args.flow.resolve(), args.adjoint.resolve())


if __name__ == "__main__":
    main()
