"""Isolation pins for the optional UI.

The interactive UI must stay one-directionally coupled: ``ui -> core``, never
the reverse, and importing the core must not pull the heavy optional ``[ui]``
dependencies.  These pins fail loudly if a future edit eagerly wires streamlit
into an engine import path.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE_DIR = REPO / "src" / "grainsmith"


def test_core_imports_do_not_pull_ui_or_streamlit():
    """A clean interpreter importing the core must NOT load grainsmith.ui or
    streamlit — the ``grainsmith ui`` handler imports them lazily, inside the
    function, so neither appears in ``sys.modules`` after a core import."""
    code = (
        "import sys\n"
        "import grainsmith, grainsmith.cli, grainsmith.pipeline\n"
        "import grainsmith.config.resolve, grainsmith.config.introspect\n"
        "bad = [m for m in ('grainsmith.ui', 'streamlit', 'matplotlib', 'pandas',\n"
        "                   'fastapi', 'uvicorn', 'starlette')\n"
        "       if m in sys.modules]\n"
        "assert not bad, 'core eagerly imported: ' + repr(bad)\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_no_core_module_imports_ui_at_top_level():
    """Static guard: outside the ui/ subtree, any reference to ``grainsmith.ui``
    must be indented (i.e. a lazy, function-local import), never a top-level
    module import."""
    offenders: list[str] = []
    for path in CORE_DIR.rglob("*.py"):
        if "ui" in path.relative_to(CORE_DIR).parts:
            continue  # the UI package may import itself freely
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.lstrip()
            if stripped.startswith(("import grainsmith.ui", "from grainsmith.ui")):
                indent = len(line) - len(stripped)
                if indent == 0:
                    rel = path.relative_to(REPO)
                    offenders.append(f"{rel}:{lineno}: {stripped}")
    assert not offenders, "top-level grainsmith.ui imports in core:\n" + "\n".join(
        offenders
    )
