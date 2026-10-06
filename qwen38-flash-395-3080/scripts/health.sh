#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import os, sys, urllib.request
host = os.environ.get("STRATA_HEALTH_HOST", "127.0.0.1")
port = os.environ.get("STRATA_HTTP_PORT", "8095")
url = "http://%s:%s/health" % (host, port)
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print("%s -> %s: %s" % (url, r.status, r.read().decode("utf-8", "replace")[:400]))
        sys.exit(0 if 200 <= r.status < 300 else 1)
except Exception as exc:
    print("%s -> ERROR: %s" % (url, exc))
    sys.exit(1)
PY
