"""``python -m grainsmith.ui`` — launch the interactive studio."""
from __future__ import annotations

import sys

from grainsmith.ui import launch

if __name__ == "__main__":
    sys.exit(launch(sys.argv[1:]))
