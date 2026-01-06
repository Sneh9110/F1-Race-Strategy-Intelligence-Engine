"""Cleanup old models from model registry.

Placeholder implementation: keep for cron wiring.
Extend to remove archived models beyond retention policy.
"""

from datetime import datetime


def main() -> int:
    print(f"[cleanup_old_models] started at {datetime.utcnow().isoformat()}Z")
    # TODO: implement retention-based cleanup in model registry
    print("[cleanup_old_models] no-op (not yet implemented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
