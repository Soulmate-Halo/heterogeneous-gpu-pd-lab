#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import os, sys, urllib.request
url = os.environ.get("STRATA_HEALTH_URL", "http://127.0.0.1:8400/health")
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print(f"{url} -> {r.status}: {r.read().decode('utf-8', 'replace')[:400]}")
        sys.exit(0 if 200 <= r.status < 300 else 1)
except Exception as e:
    print(f"{url} -> ERROR: {e}")
    sys.exit(1)
PY

