#!/usr/bin/env bash
# Delete every layer checkpoint under results/ so the next sprint starts clean.
# Usage:
#   ./reset_checkpoints.sh          # confirm, then delete
#   ./reset_checkpoints.sh -y       # no prompt
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

YES=0
if [[ "${1:-}" == "-y" || "${1:-}" == "--yes" ]]; then
  YES=1
fi

mapfile -t DIRS < <(find results -type d -name checkpoint 2>/dev/null | sort || true)

if [[ ${#DIRS[@]} -eq 0 ]]; then
  echo "no checkpoints under ${ROOT}/results"
  exit 0
fi

echo "will delete ${#DIRS[@]} checkpoint dir(s):"
printf '  %s\n' "${DIRS[@]}"

if [[ "$YES" -ne 1 ]]; then
  read -r -p "type yes to delete: " ans
  if [[ "$ans" != "yes" ]]; then
    echo "aborted"
    exit 1
  fi
fi

for d in "${DIRS[@]}"; do
  rm -rf -- "$d"
  echo "deleted $d"
done

echo "done — next go run . starts epoch 1 with no resume"
