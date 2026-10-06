#!/bin/bash
# Submit the paired state prerequisite and seven inherited generators x two rates.
# This script submits jobs; it never trains on the login node.
set -euo pipefail
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
: "${PSLG_SAMPLING_COMMIT:?set the published full commit before submission}"
PSLG_CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$PSLG_CURRENT_COMMIT" != "$PSLG_SAMPLING_COMMIT" ]] || \
   [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing submission: use the published clean checkout." >&2
  exit 2
fi
mkdir -p log
export PSLG_SAMPLING_COMMIT
PSLG_STATES_REPLY="$(sbatch --parsable --export=ALL slurm/sampling_generation_states.sbatch)"
PSLG_STATES_JOB="${PSLG_STATES_REPLY%%;*}"
[[ "$PSLG_STATES_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch reply: $PSLG_STATES_REPLY" >&2; exit 2; }
echo "state_job=$PSLG_STATES_JOB (submitted; no automatic cancellation on later submission failure)"
export PSLG_SAMPLING_STATES="/home/scnu2024024563/pslg_artifacts/sampling_states_${PSLG_STATES_JOB}"
# afterany starts the matrix after the prerequisite terminates, including a
# failed prerequisite. Each task verifies state_ready BEFORE importing a GPU
# framework or training, and packages a useful failure instead of hanging
# indefinitely with DependencyNeverSatisfied after a failed state job.
PSLG_ARRAY_REPLY="$(sbatch --parsable --export=ALL --dependency="afterany:${PSLG_STATES_JOB}" slurm/sampling_generation_array.sbatch)"
PSLG_ARRAY_JOB="${PSLG_ARRAY_REPLY%%;*}"
[[ "$PSLG_ARRAY_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch reply: $PSLG_ARRAY_REPLY" >&2; exit 2; }
echo "generation_array=$PSLG_ARRAY_JOB (14 tasks, at most one active GPU task)"
echo "Do not pull or edit this checkout until both jobs finish. Quality rejection is recorded, never retried with changed parameters."
echo "After completion, collect with:"
echo "conda run --no-capture-output -n pslg-nilm python scripts/collect_sampling_generation.py --state-job $PSLG_STATES_JOB --array-job $PSLG_ARRAY_JOB"
