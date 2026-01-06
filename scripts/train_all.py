"""Training orchestrator for all ML models.

Runs all train_*.py scripts in a deterministic order.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train_tire_degradation.py",
    "train_lap_time.py",
    "train_safety_car.py",
    "train_pit_stop_loss.py",
]


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        script_path = scripts_dir / script
        if not script_path.exists():
            print(f"[train_all] missing: {script_path}", file=sys.stderr)
            return 2

        print(f"[train_all] running: {script_path}")
        result = subprocess.run([sys.executable, str(script_path)], check=False)
        if result.returncode != 0:
            print(f"[train_all] failed: {script} rc={result.returncode}", file=sys.stderr)
            return result.returncode

    print("[train_all] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
