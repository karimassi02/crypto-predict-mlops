"""Launch the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT_DIR / "src" / "api" / "dashboard.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
         "--server.headless", "true"],
    )
