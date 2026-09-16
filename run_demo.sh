#!/usr/bin/env bash
# Reproduce the 5SPOT benchmark cases (adjoint low-rank GN and EnIF-MDA).
#
# Prerequisites (see README.md):
#   1. ./install_opm.sh       # OPM Flow + flow_adjoint in ./opm
#   2. python3.12+ on PATH    # the script creates .venv
#
# Usage:
#   ./run_demo.sh                     # all stages, fresh timestamped run id
#   ./run_demo.sh gn enif collect     # selected stages (requires RUN_ID)
#   RUN_ID=existing ./run_demo.sh collect
#
# Each invocation writes a self-contained directory case/runs/<RUN_ID> with a
# run.json manifest, both method runs and case/results-equivalent outputs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 ]]; then
  STAGES=("$@")
else
  STAGES=(prepare gn enif collect)
fi
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-$ROOT/benchmark.toml}"
RUN_DIR="$ROOT/case/runs/$RUN_ID"

OPM="$ROOT/opm"
FLOW_BIN="$OPM/opm-simulators-hnil/build/bin/flow"
ADJOINT_BIN="$OPM/opm-adjoint/build/flow_adjoint"
for binary in "$FLOW_BIN" "$ADJOINT_BIN"; do
  if [[ ! -x "$binary" ]]; then
    echo "missing $binary" >&2
    echo "build the OPM adjoint stack first: ./install_opm.sh" >&2
    exit 1
  fi
done
export LD_LIBRARY_PATH="$OPM/native/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/ert" ]]; then
  echo "== creating python environment"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install "$ROOT/packages/opm-adjoint-chainrule"
  "$VENV/bin/pip" install "$ROOT/"
else
  "$VENV/bin/pip" install --no-deps --force-reinstall -q "$ROOT/packages/opm-adjoint-chainrule" "$ROOT/"
fi
export PATH="$VENV/bin:$OPM/opm-simulators-hnil/build/bin:$PATH"

for stage in "${STAGES[@]}"; do
  case "$stage" in
    prepare)
      echo "== stage prepare: run id $RUN_ID (config: $CONFIG)"
      "$VENV/bin/opm-ert-demo-make-case" \
        --source "$ROOT/case" --output "$RUN_DIR" --config "$CONFIG" \
        --flow "$FLOW_BIN" --adjoint "$ADJOINT_BIN"
      ;;
    gn)
      echo "== stage gn: adjoint low-rank Gauss-Newton"
      "$VENV/bin/python" -m opm_ert_demo.gn_case --run-dir "$RUN_DIR"
      ;;
    enif)
      echo "== stage enif: EnIF-MDA (five weighted updates)"
      "$VENV/bin/python" -m opm_ert_demo.enif_case --run-dir "$RUN_DIR"
      ;;
    collect)
      echo "== stage collect: metrics and comparison figure"
      "$VENV/bin/opm-ert-demo-collect" --run-dir "$RUN_DIR"
      ;;
    *)
      echo "unknown stage: $stage (valid: prepare gn enif collect)" >&2
      exit 2
      ;;
  esac
done

echo "done: results in $RUN_DIR/results (manifest: $RUN_DIR/run.json)"
