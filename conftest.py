"""Project-level pytest config — adds the repo root to sys.path so that the
src/ packages can be imported as ``import src.common.grid`` etc.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
