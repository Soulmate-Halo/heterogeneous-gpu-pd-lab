#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
python3 "$ROOT/verify_bundle.py"
python3 "$ROOT/router/selftest.py"
for f in "$ROOT"/scripts/*.sh; do bash -n "$f"; done
python3 -m py_compile "$ROOT/router/router.py" "$ROOT/router/selftest.py" "$ROOT/verify_bundle.py"
echo "SMOKE_ALL_PASS"

