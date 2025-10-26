"""
Entry point for the Python port mirroring `master.R`.
"""

from pathlib import Path
import sys

# Ensure the `src` directory is on the path for local execution without installing.
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bauer_rudebusch.master import run_all

if __name__ == "__main__":
    run_all()
