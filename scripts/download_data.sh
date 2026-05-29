#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${LAGRANGIAN_SPLATS_DATA_URL:-}"
SCENE="${1:-all}"

if [[ -z "${BASE_URL}" ]]; then
  echo "Set LAGRANGIAN_SPLATS_DATA_URL to the public dataset/checkpoint base URL first." >&2
  echo "Example: LAGRANGIAN_SPLATS_DATA_URL=https://example.org/lagrangian-splats-data bash scripts/download_data.sh scalarsyn" >&2
  exit 1
fi

download_one() {
  local scene="$1"
  mkdir -p data/smoke gt

  curl -L "${BASE_URL}/data/${scene}.tar.gz" -o "/tmp/${scene}_data.tar.gz"
  tar -xzf "/tmp/${scene}_data.tar.gz" -C data/smoke

  if curl --fail -L "${BASE_URL}/gt/${scene}.tar.gz" -o "/tmp/${scene}_gt.tar.gz"; then
    tar -xzf "/tmp/${scene}_gt.tar.gz" -C gt
  else
    echo "No GT archive found for ${scene}; continuing without GT."
  fi
}

SCENES=(scalarsyn scalarreal suzanne sphere biplume)

if [[ "${SCENE}" == "all" ]]; then
  for scene in "${SCENES[@]}"; do
    download_one "${scene}"
  done
else
  found=0
  for scene in "${SCENES[@]}"; do
    if [[ "${SCENE}" == "${scene}" ]]; then
      found=1
      break
    fi
  done
  if [[ "${found}" -ne 1 ]]; then
    echo "Unknown scene: ${SCENE}" >&2
    exit 1
  fi
  download_one "${SCENE}"
fi
