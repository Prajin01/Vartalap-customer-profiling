"""Run the whole pipeline end to end, in order, stopping at the first failure.

    python run_pipeline.py            # everything (needs data/raw/online_retail_II.xlsx)
    python run_pipeline.py --from propensity   # resume from a step
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

STEPS = [
    ("ingest", "xlsx -> parquet"),
    ("build_warehouse", "DuckDB staging + star schema + data-quality report"),
    ("analytics", "10 SQL business queries"),
    ("features", "point-in-time features and labels"),
    ("segmentation", "K-Means vs GMM, named segments"),
    ("propensity", "baselines, LightGBM, calibration check, audit"),
    ("explain", "TreeSHAP, permutation importance, per-customer drivers"),
    ("profile_engine", "profile view + example profiles"),
]


def main() -> None:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    names = [s for s, _ in STEPS]
    start = 0
    if "--from" in sys.argv:
        start = names.index(sys.argv[sys.argv.index("--from") + 1])
    for name, what in STEPS[start:]:
        print(f"\n=== {name}: {what} ===", flush=True)
        t = time.time()
        if subprocess.run([sys.executable, "-m", f"src.{name}"]).returncode != 0:
            sys.exit(f"Step '{name}' failed - fix it and resume with: python run_pipeline.py --from {name}")
        print(f"--- {name} done in {time.time() - t:.0f}s", flush=True)
    print("\nPipeline complete. Now run:  python -m pytest -q")


if __name__ == "__main__":
    main()
