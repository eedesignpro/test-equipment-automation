"""The generated API map must match the drivers.

A stale map is worse than no map. An agent that reads it would draft a test
against methods that no longer exist. This test fails when any driver changes
without a regenerate.

Fix a failure by running:  .venv/Scripts/python.exe tools/gen_api_map.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools" / "gen_api_map.py"


def test_api_map_is_current():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"docs/api-map is stale. Run tools/gen_api_map.py.\n{result.stdout}\n{result.stderr}"
    )
