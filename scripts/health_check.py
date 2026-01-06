"""Cron-invoked health check.

Keeps lightweight checks for scheduler visibility.
"""

import os
import sys
from datetime import datetime

import requests


def main() -> int:
    api_url = os.environ.get("API_BASE_URL", "http://api:8000")
    url = f"{api_url}/api/v1/health/ready"
    print(f"[health_check] {datetime.utcnow().isoformat()}Z checking {url}")

    try:
        resp = requests.get(url, timeout=5)
        print(f"[health_check] status={resp.status_code} body={resp.text[:200]}")
        return 0 if resp.ok else 1
    except Exception as exc:
        print(f"[health_check] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
