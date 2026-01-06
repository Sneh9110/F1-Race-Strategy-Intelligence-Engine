#!/bin/sh
set -eu

# Load env vars if present
if [ -f /app/.env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' /app/.env | xargs) || true
fi

echo "[scheduler] starting cron (crond)" 1>&2

# dcron runs in foreground with -f; logs to stdout with -l
exec crond -f -l 8
