#!/bin/bash
# Activate each child environment, including its activation hooks. Never edit
# CUDA_VISIBLE_DEVICES: GPU visibility belongs to the Slurm allocation.
set -eo pipefail
: "${SLURM_JOB_ID:?run sampling workloads inside a Slurm allocation}"
if [[ $# -lt 3 ]]; then
  echo "usage: run_in_sampling_env.sh CONDA_INIT ENV_PYTHON SCRIPT_OR_ARGS..." >&2
  exit 2
fi
PSLG_CONDA_INIT=$1
PSLG_ENV_PYTHON=$2
shift 2
if [[ ! -f "$PSLG_CONDA_INIT" || ! -x "$PSLG_ENV_PYTHON" || "$PSLG_ENV_PYTHON" != /*/bin/python ]]; then
  echo "invalid Conda initialization script or environment Python path" >&2
  exit 2
fi
PSLG_ENV_PREFIX=${PSLG_ENV_PYTHON%/bin/python}
# Conda activation hooks may refer to unset variables; enable nounset after
# activation, as the wrapper does not need it while sourcing third-party hooks.
source "$PSLG_CONDA_INIT"
conda activate "$PSLG_ENV_PREFIX"
set -u
if [[ "${CONDA_PREFIX:-}" != "$PSLG_ENV_PREFIX" ]]; then
  echo "Conda activation did not select the requested environment" >&2
  exit 2
fi
echo "[sampling-env] conda_prefix=$CONDA_PREFIX python=$PSLG_ENV_PYTHON"
exec "$PSLG_ENV_PYTHON" "$@"
