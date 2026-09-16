#!/usr/bin/env bash
# Build the OPM Flow + flow_adjoint stack used by this demo into ./opm.
# Adapted from DEMO/BUILDING_ADJOINT_FLOW.md (see that file for details).
#
# Usage:
#   ./install_opm.sh                 # all stages in order
#   ./install_opm.sh dune opm-grid   # selected stages
#   FORCE=1 ./install_opm.sh <stage> # redo ignoring the stamp
#
# Stages are stamped in opm/.build-stamps/<stage>.done and skipped on re-run.
# Produces: opm/opm-simulators-hnil/build/bin/flow and
#           opm/opm-adjoint/build/flow_adjoint
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$ROOT/opm"
mkdir -p "$W"
cd "$W"

PARALLEL="${PARALLEL:-$(nproc)}"
STAMPS="$W/.build-stamps"
DOWNLOADS="$W/downloads"
OPM_ADJOINT_URL="${OPM_ADJOINT_URL:-https://github.com/KriFos1/opm-adjoint.git}"
mkdir -p "$STAMPS" "$DOWNLOADS"

stage_done() { [[ -f "$STAMPS/$1.done" ]]; }
mark_done()  { touch "$STAMPS/$1.done"; }

setup_env() {
  export PATH="$W/native/bin:$PATH"
  export CC="$W/native/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$W/native/bin/x86_64-conda-linux-gnu-c++"
  export FC="$W/native/bin/x86_64-conda-linux-gnu-gfortran"
  export LD_LIBRARY_PATH="$W/native/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

stage_toolchain() {
  local tarball="$DOWNLOADS/micromamba.tar.bz2"
  if [[ ! -x "$W/bin/micromamba" ]]; then
    if [[ ! -f "$tarball" ]]; then
      curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$tarball"
    fi
    tar -xjf "$tarball" -C "$W" bin/micromamba
  fi
  if [[ ! -d "$W/native" ]]; then
    "$W/bin/micromamba" create -y -p "$W/native" \
      gcc_linux-64=16.2 gxx_linux-64=16.2 gfortran_linux-64=16.2 \
      cmake=4.4.3 boost-cpp=1.85 fmt=11.2 suitesparse=7 \
      python=3.12 'liblapack=3.11=*openblas'
  fi
  setup_env
}

stage_clone() {
  if [[ ! -d opm-common-hnil/.git ]]; then
    git clone https://github.com/hnil/opm-common.git opm-common-hnil
  fi
  git -C opm-common-hnil checkout 0ebb110

  if [[ ! -d opm-grid-hnil/.git ]]; then
    git clone https://github.com/hnil/opm-grid.git opm-grid-hnil
  fi
  git -C opm-grid-hnil checkout 8cb1da3

  if [[ ! -d opm-simulators-hnil/.git ]]; then
    git clone https://github.com/hnil/opm-simulators.git opm-simulators-hnil
  fi
  git -C opm-simulators-hnil fetch origin adjoint-hooks
  git -C opm-simulators-hnil checkout 9b07194

  if [[ ! -d opm-adjoint/.git ]]; then
    git clone -b feat/matchw-linw-objectives "$OPM_ADJOINT_URL" opm-adjoint
  fi
  git -C opm-adjoint checkout 8599b5a

  if [[ ! -d opm-tests-adjoint/.git ]]; then
    git clone -b add_adjoint_tests https://github.com/hnil/opm-tests.git opm-tests-adjoint
  fi
  git -C opm-tests-adjoint checkout 03c0b39

  mkdir -p deps
  local m
  for m in dune-common dune-geometry dune-istl dune-grid; do
    if [[ ! -d deps/$m/.git ]]; then
      git clone -b releases/2.10 "https://gitlab.dune-project.org/core/$m.git" deps/$m
    fi
  done
  git -C deps/dune-common checkout 2988ac1
  git -C deps/dune-grid checkout 954436b
  if [[ ! -f "$STAMPS/dune-pins.env" ]]; then
    {
      echo "DUNE_GEOMETRY_PIN=$(git -C deps/dune-geometry rev-parse HEAD)"
      echo "DUNE_ISTL_PIN=$(git -C deps/dune-istl rev-parse HEAD)"
    } > "$STAMPS/dune-pins.env"
  fi
  # shellcheck disable=SC1091
  source "$STAMPS/dune-pins.env"
  git -C deps/dune-geometry checkout "$DUNE_GEOMETRY_PIN"
  git -C deps/dune-istl checkout "$DUNE_ISTL_PIN"
}

stage_patch() {
  local p="$STAMPS/opmsatellites-guard.patch"
  cat > "$p" <<'EOF'
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
EOF
  if git -C opm-common-hnil apply --check --reverse "$p" >/dev/null 2>&1; then
    echo "opm-common patch already applied"
    return 0
  fi
  git -C opm-common-hnil apply "$p" ||
    patch -N -p1 -d opm-common-hnil --fuzz=3 < "$p"
}

stage_dune() {
  setup_env
  local m
  for m in dune-common dune-geometry dune-istl dune-grid; do
    cmake -S "$W/deps/$m" -B "$W/deps/$m/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$W/deps/install" "-DCMAKE_PREFIX_PATH=$W/native" \
      -DBUILD_TESTING=OFF -DDUNE_ENABLE_PYTHONBINDINGS=OFF \
      -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON \
      -DCMAKE_DISABLE_FIND_PACKAGE_QuadMath=ON \
      -DCMAKE_DISABLE_FIND_PACKAGE_Vc=ON
    cmake --build "$W/deps/$m/build" --parallel "$PARALLEL" --target install
  done
}

stage_opm_common() {
  setup_env
  cmake -S "$W/opm-common-hnil" -B "$W/opm-common-hnil/build" -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_PREFIX_PATH=$W/deps/install;$W/native" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=ON -DENABLE_MOCKSIM=OFF \
    -DOPM_ENABLE_PYTHON=OFF -DOPM_INSTALL_PYTHON=OFF -DOPM_ENABLE_EMBEDDED_PYTHON=OFF \
    -DOPM_ENABLE_DUNE=ON -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
    -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE -DCMAKE_DISABLE_FIND_PACKAGE_QuadMath=ON
  cmake --build "$W/opm-common-hnil/build" --parallel "$PARALLEL"
}

stage_opm_grid() {
  setup_env
  cmake -S "$W/opm-grid-hnil" -B "$W/opm-grid-hnil/build" -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_PREFIX_PATH=$W/opm-common-hnil/build;$W/deps/install;$W/native" \
    -Dopm-common_DIR="$W/opm-common-hnil/build" \
    -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
    -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE \
    -DCMAKE_DISABLE_FIND_PACKAGE_dune-uggrid=ON -DCMAKE_DISABLE_FIND_PACKAGE_ZOLTAN=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_METIS=ON
  cmake --build "$W/opm-grid-hnil/build" --parallel "$PARALLEL"
}

stage_opm_simulators() {
  setup_env
  cmake -S "$W/opm-simulators-hnil" -B "$W/opm-simulators-hnil/build" -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_PREFIX_PATH=$W/opm-common-hnil/build;$W/opm-grid-hnil/build;$W/deps/install;$W/native" \
    -Dopm-common_DIR="$W/opm-common-hnil/build" -Dopm-grid_DIR="$W/opm-grid-hnil/build" \
    -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_FLOW=ON -DBUILD_FLOW_VARIANTS=OFF \
    -DUSE_MPI=OFF -DUSE_OPENMP=OFF -DUSE_GPU_BRIDGE=OFF -DUSE_OPENCL=OFF -DUSE_AMGX=OFF \
    -DUSE_HYPRE=OFF -DUSE_DAMARIS_LIB=OFF -DOPM_ENABLE_PYTHON=OFF -DOPM_INSTALL_PYTHON=OFF \
    -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON -DCMAKE_DISABLE_FIND_PACKAGE_QuadMath=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_dune-alugrid=ON -DCMAKE_DISABLE_FIND_PACKAGE_dune-fem=ON
  cmake --build "$W/opm-simulators-hnil/build" --parallel "$PARALLEL" --target flow opmsimulators
}

stage_opm_adjoint() {
  setup_env
  cmake -S "$W/opm-adjoint" -B "$W/opm-adjoint/build" -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_PREFIX_PATH=$W/opm-common-hnil/build;$W/opm-grid-hnil/build;$W/opm-simulators-hnil/build;$W/deps/install;$W/native" \
    -Dopm-common_DIR="$W/opm-common-hnil/build" -Dopm-grid_DIR="$W/opm-grid-hnil/build" \
    -Dopm-simulators_DIR="$W/opm-simulators-hnil/build" \
    -DBUILD_TESTING=ON -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
    -DOPM_INTERPROCEDURAL_OPTIMIZATION_TYPE=NONE -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON
  # Known upstream issue on this branch: tests/test_transposeMatrix fails to
  # link (wants FlexibleSolver<FieldMatrix> instantiations; libopmsimulators
  # only provides Opm::MatrixBlock flavors). flow_adjoint itself is unaffected.
  cmake --build "$W/opm-adjoint/build" --parallel "$PARALLEL" ||
    { echo "note: full build failed on the known test link error; building flow_adjoint target only"
      cmake --build "$W/opm-adjoint/build" --parallel "$PARALLEL" --target flow_adjoint; }
}

stage_verify() {
  setup_env
  test -x "$W/opm-simulators-hnil/build/bin/flow"
  test -x "$W/opm-adjoint/build/flow_adjoint"
  "$W/opm-simulators-hnil/build/bin/flow" --version
  "$W/opm-adjoint/build/flow_adjoint" --version
}

stage_smoke() {
  setup_env
  local d="$W/smoke/MODEL_1D_DEBUG"
  mkdir -p "$d"
  cp "$W"/opm-tests-adjoint/adjoint_tests/MODEL_1D_DEBUG/inputfiles/* "$d"/
  # adjoint-hooks deck constraints: WCONPROD/WCONINJE VFP=-1 is rejected
  sed -i 's/ -1 / 0 /' "$d/wconprodstep_1.txt" "$d/wconinjestep_1.txt"
  cat > "$d/linw.txt" <<'EOF'
2011-11-06 Inj WATER -86400.0
2011-11-06 Prod OIL -86400.0
2011-12-06 Inj WATER -86400.0
2011-12-06 Prod OIL -86400.0
2012-01-06 Inj WATER -86400.0
2012-01-06 Prod OIL -86400.0
2012-02-05 Inj WATER -86400.0
2012-02-05 Prod OIL -86400.0
2012-03-07 Inj WATER -86400.0
2012-03-07 Prod OIL -86400.0
2012-04-06 Inj WATER -86400.0
2012-04-06 Prod OIL -86400.0
2012-05-06 Inj WATER -86400.0
2012-05-06 Prod OIL -86400.0
EOF
  rm -rf "$d/run0" "$d/sweep0"
  local strict=(--enable-storage-cache=false --tolerance-cnv=1e-7
    --tolerance-cnv-relaxed=1e-6 --tolerance-mb=1e-9
    --tolerance-mb-relaxed=1e-9 --newton-max-iterations=40)
  (cd "$d" && "$W/opm-adjoint/build/flow_adjoint" MODEL_1D_DEBUG.DATA \
      --output-dir=run0 "${strict[@]}" \
      --adjoint-file=run0/archive --adjoint-save=true > record.log 2>&1)
  test -f "$d/run0/archive/adjoint/meta.bin"
  (cd "$d" && "$W/opm-adjoint/build/flow_adjoint" MODEL_1D_DEBUG.DATA \
      --output-dir=sweep0 "${strict[@]}" \
      --adjoint-file=run0/archive --adjoint-mode=gradient \
      --adjoint-objective='linw:linw.txt' > replay.log 2>&1)
  local nz
  for f in PV PERM; do
    nz=$(awk '$1+0!=0 {c++} END{print c+0}' "$d/sweep0/MODEL_1D_DEBUG.ADJOINT_GRADIENTS_$f.txt")
    echo "ADJOINT_GRADIENTS_$f: nonzero rows = $nz"
    [[ "$nz" -gt 0 ]]
  done
}

STAGES=(toolchain clone patch dune opm-common opm-grid opm-simulators opm-adjoint verify smoke)

main() {
  local requested=("${@:-${STAGES[@]}}")
  local s fn
  for s in "${requested[@]}"; do
    fn="stage_${s//-/_}"
    if ! declare -f "$fn" >/dev/null; then
      echo "unknown stage: $s (valid: ${STAGES[*]})" >&2
      exit 2
    fi
    if [[ "$s" != toolchain && ! -d "$W/native" ]]; then
      echo "stage $s requires the toolchain; run './install_opm.sh toolchain' first" >&2
      exit 1
    fi
    if stage_done "$s" && [[ "${FORCE:-0}" != 1 ]]; then
      echo "== stage $s: already done (.build-stamps/$s.done; FORCE=1 to redo)"
      continue
    fi
    echo "== stage $s: start $(date -Is)"
    "$fn"
    mark_done "$s"
    echo "== stage $s: OK $(date -Is)"
  done
}

main "$@"
