#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
bash "$ROOT/entrypoint.sh" help >/dev/null
bash "$ROOT/entrypoint.sh" doctor
python3 "$ROOT/verify_bundle.py"
for role in spark 3080; do
    set +e
    output="$(bash "$ROOT/entrypoint.sh" "$role" 2>&1)"; rc=$?
    set -e
    [[ $rc == 2 ]] || { echo "FAIL missing-asset role=$role exit=$rc" >&2; exit 1; }
    [[ "$output" == *STRATA_DENSE_GGUF* && "$output" == *'not included'* ]] || { echo "FAIL asset guidance role=$role" >&2; exit 1; }
done
python3 "$ROOT/router/selftest.py" >/dev/null 2>&1
echo CONTAINER_SMOKE_PASS
