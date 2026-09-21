#!/bin/bash
# Source this file after cloning the repository. It derives the project root
# from its own location, so the clone directory may be PSLG-NILM, PSLG-NILM-c1,
# or another user-chosen name.

_pslg_paths_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PSLG_PROJECT_ROOT="${PSLG_PROJECT_ROOT:-$(cd "${_pslg_paths_dir}/.." && pwd)}"
export PSLG_DATA_ROOT="${PSLG_DATA_ROOT:-$HOME/pslg_data}"
export PSLG_ARTIFACT_ROOT="${PSLG_ARTIFACT_ROOT:-$HOME/pslg_artifacts}"
export PSLG_LOG_ROOT="${PSLG_LOG_ROOT:-$HOME/pslg_logs}"
export PSLG_MANIFEST_ROOT="${PSLG_MANIFEST_ROOT:-$HOME/pslg_manifests}"
export PSLG_REGISTRY_ROOT="${PSLG_REGISTRY_ROOT:-$HOME/pslg_registries}"

mkdir -p "$PSLG_DATA_ROOT" "$PSLG_ARTIFACT_ROOT" "$PSLG_LOG_ROOT" \
  "$PSLG_MANIFEST_ROOT" "$PSLG_REGISTRY_ROOT"

unset _pslg_paths_dir

printf '[pslg] project=%s\n' "$PSLG_PROJECT_ROOT"
printf '[pslg] data=%s\n' "$PSLG_DATA_ROOT"
printf '[pslg] artifacts=%s\n' "$PSLG_ARTIFACT_ROOT"
printf '[pslg] logs=%s\n' "$PSLG_LOG_ROOT"
printf '[pslg] manifests=%s\n' "$PSLG_MANIFEST_ROOT"
printf '[pslg] registries=%s\n' "$PSLG_REGISTRY_ROOT"
