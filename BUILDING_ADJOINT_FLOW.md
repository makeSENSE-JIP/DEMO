# Building OPM Flow with adjoint support

This runbook rebuilds the DROGON_TEST2_OPM adjoint stack from scratch on a
clean Linux box. It produces two binaries:

- `flow` — plain forward simulator (opm-simulators)
- `flow_adjoint` — record/replay/gradient build (opm-adjoint) with the
  `matchw:`/`linw:` weighted well-rate objectives

Everything builds without MPI, OpenMP, or GPU support. Serial, one thread
per process, is what the adjoint replay path has been validated with.

## Repository pins

| Repository | Ref | Commit | Role |
|---|---|---|---|
| `hnil/opm-common` | master | `0ebb110` | parser, io |
| `hnil/opm-grid` | master | `8cb1da3` | grid library |
| `hnil/opm-simulators` | branch `adjoint-hooks` | `9b07194` | flow + replay hooks |
| `KriFos1/opm-adjoint` | branch `feat/matchw-linw-objectives` | `8599b5a` (base: `hnil/opm-adjoint` master `c9b7067`) | adjoint driver + new objectives |
| `hnil/opm-tests` | branch `add_adjoint_tests` | `03c0b39` | fixture decks (optional) |
| dune-common / dune-geometry / dune-istl / dune-grid | `releases/2.10` | dune-common `2988ac1`, dune-grid `954436b` (istl/geometry: branch tip) | numerics |

`opm-adjoint` master from hnil lacks the weighted objectives. The feature
branch in `KriFos1/opm-adjoint` carries them; hnil master is the merge base.
All other repositories are used exactly as published, with one local patch
to opm-common described below.

## Toolchain

The validated toolchain is a conda-forge prefix (created with micromamba)
plus a Python venv for the opm python bindings:

```bash
micromamba create -y -p ./native \
  gcc_linux-64=16.2 gxx_linux-64=16.2 gfortran_linux-64=16.2 \
  cmake=4.4.3 boost-cpp=1.85 fmt=11.2 \
  liblapack=3.11=*openblas
python3.12 -m venv .venv
```

Export these in every shell that configures or builds:

```bash
W=$PWD
export PATH=$W/native/bin:$PATH
export CC=$W/native/bin/x86_64-conda-linux-gnu-cc
export CXX=$W/native/bin/x86_64-conda-linux-gnu-c++
export FC=$W/native/bin/x86_64-conda-linux-gnu-gfortran
export LD_LIBRARY_PATH=$W/native/lib:${LD_LIBRARY_PATH:-}
```

Any GCC with C++20 support and CMake >= 3.28 works in principle; the pins
above are what the gradients were verified with.

## Clone and pin

```bash
git clone https://github.com/hnil/opm-common.git  opm-common-hnil
git -C opm-common-hnil checkout 0ebb110
git clone https://github.com/hnil/opm-grid.git    opm-grid-hnil
git -C opm-grid-hnil checkout 8cb1da3
git clone https://github.com/hnil/opm-simulators.git opm-simulators-hnil
git -C opm-simulators-hnil fetch origin adjoint-hooks
git -C opm-simulators-hnil checkout 9b07194
git clone -b feat/matchw-linw-objectives git@github.com:KriFos1/opm-adjoint.git
# optional fixtures:
git clone -b add_adjoint_tests https://github.com/hnil/opm-tests.git opm-tests-adjoint

mkdir -p deps && cd deps
for m in dune-common dune-geometry dune-istl dune-grid; do
  git clone -b releases/2.10 https://gitlab.dune-project.org/core/$m.git
done
git -C dune-common checkout 2988ac1
git -C dune-grid checkout 954436b
cd ..
```

## Required patch to opm-common

`opm-common-hnil` master references test targets unconditionally in
`cmake/Modules/OpmSatellites.cmake`, which breaks configuring with
`-DBUILD_TESTING=OFF`. Keep this as a local working-tree patch:

```diff
--- a/cmake/Modules/OpmSatellites.cmake
+++ b/cmake/Modules/OpmSatellites.cmake
@@ -343,7 +343,9 @@
     if(NOT TARGET test-suite)
       add_custom_target(test-suite)
     endif()
-    add_dependencies(test-suite ${CURTEST_EXE_TARGET})
+    if(TARGET ${CURTEST_EXE_TARGET})
+      add_dependencies(test-suite ${CURTEST_EXE_TARGET})
+    endif()
   endif()
 endfunction()
```

Alternatively configure opm-common with `BUILD_TESTING=ON`; then the guard
is unnecessary but the build pulls extra test dependencies.

## Build order

Build dune first, then the OPM chain strictly in order; each stage points
`CMAKE_PREFIX_PATH` at the build trees before it. Nothing installs into
system paths. `--parallel 8` is a knob; adjust to the machine.

### 1. dune 2.10 (installed into deps/install)

```bash
for m in dune-common dune-geometry dune-istl dune-grid; do
  cmake -S $W/deps/$m -B $W/deps/$m/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$W/deps/install "-DCMAKE_PREFIX_PATH=$W/native" \
    -DBUILD_TESTING=OFF -DDUNE_ENABLE_PYTHONBINDINGS=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_QuadMath=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_Vc=ON
  cmake --build $W/deps/$m/build --parallel 8 --target install
done
```

### 2. opm-common

`BUILD_EXAMPLES=ON` is not optional: opm-simulators consumes the
`rst_deck` example target. cJSON is FetchContent'd at configure time, so
this step needs network access to github.com (on a box without it,
pre-populate a cjson checkout and add
`-DFETCHCONTENT_SOURCE_DIR_CJSON=<path>/cjson-src`).

```bash
cmake -S $W/opm-common-hnil -B $W/opm-common-hnil/build -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$W/deps/install;$W/native" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=ON -DENABLE_MOCKSIM=OFF \
  -DOPM_ENABLE_PYTHON=ON -DOPM_INSTALL_PYTHON=OFF -DOPM_ENABLE_EMBEDDED_PYTHON=OFF \
  -DOPM_ENABLE_DUNE=ON -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
  -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE -DCMAKE_DISABLE_FIND_PACKAGE_QuadMath=ON \
  -DPython3_EXECUTABLE=$W/.venv/bin/python
cmake --build $W/opm-common-hnil/build --parallel 8
```

Set `OPM_ENABLE_PYTHON=OFF` (and drop `Python3_EXECUTABLE`) if the python
bindings for EGRID/ESmry access are not needed.

### 3. opm-grid

```bash
cmake -S $W/opm-grid-hnil -B $W/opm-grid-hnil/build -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$W/opm-common-hnil/build;$W/deps/install;$W/native" \
  -Dopm-common_DIR=$W/opm-common-hnil/build \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
  -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE \
  -DCMAKE_DISABLE_FIND_PACKAGE_dune-uggrid=ON -DCMAKE_DISABLE_FIND_PACKAGE_ZOLTAN=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_METIS=ON
cmake --build $W/opm-grid-hnil/build --parallel 8
```

### 4. opm-simulators (flow + the library opm-adjoint links)

```bash
cmake -S $W/opm-simulators-hnil -B $W/opm-simulators-hnil/build -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$W/opm-common-hnil/build;$W/opm-grid-hnil/build;$W/deps/install;$W/native" \
  -Dopm-common_DIR=$W/opm-common-hnil/build -Dopm-grid_DIR=$W/opm-grid-hnil/build \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_FLOW=ON -DBUILD_FLOW_VARIANTS=OFF \
  -DUSE_MPI=OFF -DUSE_OPENMP=OFF -DUSE_GPU_BRIDGE=OFF -DUSE_OPENCL=OFF -DUSE_AMGX=OFF \
  -DUSE_HYPRE=OFF -DUSE_DAMARIS_LIB=OFF -DOPM_ENABLE_PYTHON=OFF -DOPM_INSTALL_PYTHON=OFF \
  -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON -DCMAKE_DISABLE_FIND_PACKAGE_QuadMath=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_dune-alugrid=ON -DCMAKE_DISABLE_FIND_PACKAGE_dune-fem=ON
cmake --build $W/opm-simulators-hnil/build --parallel 8 --target flow opmsimulators
```

### 5. opm-adjoint (flow_adjoint)

```bash
cmake -S $W/opm-adjoint -B $W/opm-adjoint/build -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$W/opm-common-hnil/build;$W/opm-grid-hnil/build;$W/opm-simulators-hnil/build;$W/deps/install;$W/native" \
  -Dopm-common_DIR=$W/opm-common-hnil/build -Dopm-grid_DIR=$W/opm-grid-hnil/build \
  -Dopm-simulators_DIR=$W/opm-simulators-hnil/build \
  -DBUILD_TESTING=ON -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
  -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON
cmake --build $W/opm-adjoint/build --parallel 8
```

Binaries land in `opm-simulators-hnil/build/bin/flow` and
`opm-adjoint/build/flow_adjoint`.

## Runtime environment

```bash
export LD_LIBRARY_PATH=$W/native/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$W/opm-common-hnil/build/python:$PYTHONPATH   # optional, opm.io EGrid/ESmry
```

`flow_adjoint` needs the conda `libstdc++`/libs from `native/lib` at
runtime; without it the binary aborts on library lookup.

## Using the adjoint build

Forward run with trajectory recording (archive written last; `meta.bin`
inside signals completion):

```bash
flow_adjoint CASE.DATA --output-dir=run0 \
  --enable-storage-cache=false --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-6 \
  --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 --newton-max-iterations=40 \
  --adjoint-file=run0/archive --adjoint-save=true
```

Gradient/VJP replay against the recorded archive, writing
`CASE.ADJOINT_GRADIENTS_{PV,PERM,TRANS,WELLCTRL}.txt`:

```bash
flow_adjoint CASE.DATA --output-dir=sweep0 \
  --enable-storage-cache=false <same strict tolerances> \
  --adjoint-file=run0/archive --adjoint-mode=gradient \
  --adjoint-objective='matchw:matchw.txt'
```

Objective specs (feature branch):

- `matchw:<file>` — rows `YYYY-MM-DD WELL PHASE OBS WEIGHT`; OBS/WEIGHT in
  sm3/day (producers positive); adds `0.5*w*(q-obs)^2` at substeps ending
  on a row date (0.25 d tolerance). Data-misfit term for GN gradients.
- `linw:<file>` — rows `YYYY-MM-DD WELL PHASE COEF` with COEF in internal
  units (for producer rate q in sm3/day use COEF = -86400 * w). Linear rate
  form for VJPs against arbitrary weight vectors.

Both write gradient files per substep-accumulated sums; PV is per active
cell (dJ/dpvmult), PERM is dJ/dPERMX,Y,Z in SI (per m^2; multiply by
9.869233e-16 for per-mD).

Strict tolerances and `--enable-storage-cache=false` are required for
replay-exact adjoints; adaptive time stepping stays ON for the Drogon
schedule. Keep exactly one recorded-forward or replay process per host at
a time (replay RSS can reach ~30 GB). Gradient runs truncate summary files
in their `--output-dir` at startup, so keep reference ESmry files outside
replay output directories and use a fresh output dir per sweep.

Deck constraints on the adjoint-hooks branch: WCONINJE VFP=-1 is rejected
(patch decks to 0), and multisegment wells (WELSEGS/COMPSEGS/WSEGVALV and
related) throw in the well-objective adjoint path; strip those keywords
from deck copies. Archive sizes run ~5 GB (truncated history) to ~12 GB
(full history) per recorded forward.

## Verify the build

Smoke test with the fixture deck from `opm-tests-adjoint`:

```bash
cd opm-tests-adjoint/adjoint_tests/MODEL_1D_DEBUG/inputfiles
# record
flow_adjoint MODEL_1D_DEBUG.DATA --output-dir=run0 <strict flags> \
  --adjoint-file=run0/archive --adjoint-save=true
# replay-gradient with a constant-coefficient linw file
flow_adjoint MODEL_1D_DEBUG.DATA --output-dir=sweep0 <strict flags> \
  --adjoint-file=run0/archive --adjoint-mode=gradient \
  --adjoint-objective='linw:linw.txt'
```

Nonzero finite values in `MODEL_1D_DEBUG.ADJOINT_GRADIENTS_{PV,PERM}.txt`
confirm the chain. For end-to-end verification including finite-difference
checks of the new objectives, the ERT integration ships an opt-in test:
`tests/ert/unit_tests/run_models/test_low_rank_gn_opm.py` in the
`feat/low-rank-gn-sampler` branch of the ert checkout, with
`OPM_ADJOINT_EXECUTABLE` and `OPM_ADJOINT_TEST_INPUT` pointing at this
build and the fixture above.

## Provenance summary

- Everything comes from published refs except: (a) the
  `feat/matchw-linw-objectives` branch of `KriFos1/opm-adjoint`
  (matchw/linw objectives, closed-well and schedule guards,
  case-insensitive phase parsing), and (b) the three-line
  `OpmSatellites.cmake` guard kept as a local patch in opm-common.
- The reference build of this exact stack was validated by FD checks
  (PORO/log-PERMX, median rel <= 1e-2) and by matchw replay agreement with
  an independent Python misfit evaluation to rel 2e-8 on the truncated
  Drogon case.
