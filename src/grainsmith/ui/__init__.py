"""grainsmith.ui — optional interactive UI ("grainsmith studio").

The studio is a **React single-page app** (built with Vite and bundled as static
assets) served by a small **FastAPI / uvicorn** backend.  It (a) builds
*validated* YAML configs straight from the pydantic schema, (b) runs the pipeline
in-app with live progress + log streaming, and (c) visualises a finished run's
recorded outputs.  It is deliberately **decoupled** from the engine:

* the core never imports ``grainsmith.ui`` at module load — pinned by
  ``tests/test_ui_isolation.py``;
* its optional dependencies (fastapi, uvicorn) are imported lazily and are not
  declared by this release, so plain ``import grainsmith`` (and every
  HPC/headless code path) never pulls them in.

Importing *this* module is cheap and dependency-free; the FastAPI app and the
uvicorn server are only touched inside :func:`launch`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP_TITLE = "grainsmith studio"

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765

# The built React bundle lands here (produced by `vite build` against the
# frontend source).  It ships inside the release wheel; in a plain
# source checkout it is absent until the frontend is built.
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _parse_argv(argv: list[str] | None) -> tuple[str, int, bool]:
    """Parse the launcher's own flags out of forwarded CLI tokens.

    Recognises ``--host H`` / ``--host=H``, ``--port N`` / ``--port=N`` and
    ``--no-browser``; any other token is ignored (the studio takes no further
    runtime options).
    """
    host, port, open_browser = _DEFAULT_HOST, _DEFAULT_PORT, True
    tokens = list(argv or [])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--no-browser":
            open_browser = False
        elif tok == "--host" and i + 1 < len(tokens):
            host = tokens[i + 1]
            i += 1
        elif tok.startswith("--host="):
            host = tok.split("=", 1)[1]
        elif tok == "--port" and i + 1 < len(tokens):
            port = int(tokens[i + 1])
            i += 1
        elif tok.startswith("--port="):
            port = int(tok.split("=", 1)[1])
        i += 1
    return host, port, open_browser


def launch(argv: list[str] | None = None) -> int:
    """Launch *grainsmith studio*: serve the React bundle via uvicorn.

    Binds to localhost only and opens a browser tab (unless ``--no-browser``).
    Prints a clean install hint and returns ``1`` (not a raw ``ImportError``)
    when fastapi/uvicorn are missing.  Blocks until interrupted, then
    returns ``0``.
    """
    if importlib.util.find_spec("uvicorn") is None or (
        importlib.util.find_spec("fastapi") is None
    ):
        sys.stderr.write(
            "The interactive grainsmith UI is not included in this release; "
            "it is planned for a future version.\n"
        )
        return 1

    host, port, open_browser = _parse_argv(argv)

    if not (STATIC_DIR / "index.html").is_file():
        sys.stderr.write(
            "Note: the React frontend bundle was not found at\n"
            f"    {STATIC_DIR}\n"
            "The JSON API will be live but the page shows a build hint instead.\n"
            "Build it with:  cd frontend && npm install && npm run build\n"
            "(release wheels ship the bundle inside them).\n"
        )

    import threading
    import webbrowser

    import uvicorn

    # Never advertise a 0.0.0.0 bind to the browser — open the loopback address.
    browse_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{browse_host}:{port}/"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    sys.stderr.write(f"{APP_TITLE} -> {url}  (Ctrl-C to stop)\n")
    uvicorn.run(
        "grainsmith.ui.server:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
    )
    return 0
