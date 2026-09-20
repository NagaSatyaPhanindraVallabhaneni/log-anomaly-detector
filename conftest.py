"""Put src/ on sys.path so tests import the package without installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
