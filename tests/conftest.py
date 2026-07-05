"""Shared pytest configuration for the test suite.

Ensures the repository root is importable so tests can use absolute imports
(``from src.preprocessing import ...``) regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mirror the production/CI interpreter path (PYTHONPATH=src) so packages that
# use flat internal imports (``from mlops... import ...``) resolve identically
# in local runs, CI, and the container.
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
