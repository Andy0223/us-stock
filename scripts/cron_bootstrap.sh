#!/usr/bin/env bash

# Shared runtime bootstrap for cron-launched jobs.
# Source this file from shell entrypoints; it defines PROJECT_ROOT and run_python.

unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export US_STOCK_HOME="${US_STOCK_HOME:-/Users/andy}"
export HOME="${US_STOCK_HOME}"
export LANG="${LANG:-zh_TW.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$BOOTSTRAP_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"
export PWD="$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/outputs/cron" "$PROJECT_ROOT/data/inbox/firstrade"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

PYTHON_ARCH_PREFIX=()
if [[ "${PYTHON_FORCE_X86_64:-1}" == "1" && "$(uname -s)" == "Darwin" && -x "/usr/bin/arch" ]]; then
  PYTHON_ARCH_PREFIX=(/usr/bin/arch -x86_64)
fi

run_python() {
  "${PYTHON_ARCH_PREFIX[@]}" "$PYTHON_BIN" "$@"
}

cron_python_diagnostics() {
  echo "Python: $PYTHON_BIN"
  echo "Working directory: $(pwd -P)"
  run_python -B - <<'PY'
import os
import platform
import sys

print(f"Python executable: {sys.executable}")
print(f"Python cwd: {os.getcwd()}")
print(f"Python machine: {platform.machine()}")
PY
}
