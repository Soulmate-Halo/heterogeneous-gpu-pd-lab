#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
bash "$ROOT/entrypoint.sh" help >/dev/null
python3 "$ROOT/scripts/doctor.py"
python3 "$ROOT/verify_bundle.py"
set +e
output="$(bash "$ROOT/entrypoint.sh" engine 2>&1)"; rc=$?
set -e
[[ $rc == 2 ]] || { echo "FAIL missing-asset engine exit=$rc" >&2; exit 1; }
[[ "$output" == *STRATA_DENSE_GGUF* && "$output" == *'not included'* ]] || { echo "FAIL asset guidance" >&2; exit 1; }
echo CONTAINER_SMOKE_PASS
