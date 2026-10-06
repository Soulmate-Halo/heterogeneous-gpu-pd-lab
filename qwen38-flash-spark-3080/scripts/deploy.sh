#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
case "${1:-}" in spark) PROFILE=spark; SERVICE=spark ;; 3080|engine) PROFILE=3080; SERVICE=gpu3080 ;; router) PROFILE=router; SERVICE=router ;; *) echo 'usage: deploy.sh spark|3080|router' >&2; exit 2 ;; esac
cd "$ROOT"
[[ -f .env ]] || { echo 'copy env.example to .env and fill your model directory and peer addresses' >&2; exit 2; }
docker compose --env-file .env --profile "$PROFILE" config --quiet
docker compose --env-file .env --profile "$PROFILE" pull "$SERVICE"
docker compose --env-file .env --profile "$PROFILE" up -d "$SERVICE"
docker compose --env-file .env ps "$SERVICE"
echo "Logs: docker compose --env-file .env logs -f $SERVICE"
