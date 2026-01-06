"""Cleanup old data artifacts.

Placeholder implementation: keep for cron wiring.
Extend to prune old rows/files based on retention policy.
"""

from datetime import datetime


def main() -> int:
    print(f"[cleanup_old_data] started at {datetime.utcnow().isoformat()}Z")
    # TODO: implement retention-based cleanup (DB + filesystem)
    print("[cleanup_old_data] no-op (not yet implemented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
