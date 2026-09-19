"""FastAPI backend for *grainsmith studio* (the React UI's server).

Imported only when the optional ``[ui]`` extra is installed — by the
``grainsmith ui`` launcher or by the UI tests.  The engine never imports this
module (pinned by ``tests/test_ui_isolation.py``), so a plain ``import
grainsmith`` stays free of FastAPI.

This is a THIN presentation seam: every endpoint delegates to a pure,
unit-tested module, so the UI can never fork a second config / validation /
analysis code path —

* ``GET  /api/health``            -> liveness + version
* ``GET  /api/schema``            -> :func:`grainsmith.ui.schema_json.schema_to_json`
* ``GET  /api/presets``           -> :func:`grainsmith.ui.presets.discover_presets`
* ``GET  /api/presets/{name}``    -> :func:`grainsmith.ui.presets.load_preset`
* ``POST /api/validate``          -> :func:`grainsmith.ui.yaml_builder.to_yaml`
* ``POST /api/yaml/from-values``  -> :func:`grainsmith.ui.yaml_builder.to_yaml`
* ``POST /api/yaml/to-values``    -> :func:`grainsmith.ui.yaml_builder.load_yaml`
* ``GET  /api/outputs``           -> :func:`grainsmith.ui.outputs_json.load_outputs_json`
* ``GET  /api/pick-dir``          -> native directory picker (local desktop use)
* ``GET  /api/run/notice``        -> run-capability notice for the client
* ``POST /api/run``               -> :mod:`grainsmith.ui.run_stream` (SSE
  streaming run: live progress + log output)
* ``GET  /``                      -> static React SPA bundle

A :class:`~grainsmith.errors.ConfigError` (the engine's own validation error)
maps to HTTP 422 with the original message — exactly what the CLI prints — so a
config that validates here is one the CLI accepts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import grainsmith
from grainsmith.errors import ConfigError
from grainsmith.ui import outputs_json, schema_json
from grainsmith.ui.presets import discover_presets, load_preset
from grainsmith.ui.run_stream import RUN_LOCK, RUN_MEMORY_NOTICE, run_events
from grainsmith.ui.yaml_builder import (
    config_from_values,
    config_to_yaml,
    load_yaml,
    to_yaml,
)

# The built React bundle lands here; absent until the frontend is built.
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _native_pick_directory() -> str:
    """Open a native OS "choose folder" dialog and return the chosen absolute
    path (``''`` if cancelled or unavailable).

    The studio is a localhost tool — the uvicorn server runs on the *user's own
    machine* — so a native dialog here is the natural way to point the analysis
    viewer at an output directory and get its real absolute path (a browser
    folder picker only exposes the folder name, which the server cannot read).
    The dialog runs in a short-lived subprocess so it never touches uvicorn's
    event loop / threads; if Tk or a display is unavailable, it degrades to
    ``''`` and the UI falls back to manual path entry.
    """
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "p = filedialog.askdirectory(title='Select a grainsmith output directory')\n"
        "print(p or '')\n"
    )
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        # Suppress the helper interpreter's console window (the Tk dialog still
        # shows). getattr keeps this importable/typed off-Windows too.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip()


class ValuesRequest(BaseModel):
    """Body for endpoints that take collected form values."""

    values: dict[str, Any]


class YamlRequest(BaseModel):
    """Body for parsing pasted / uploaded YAML text."""

    yaml: str


class RunRequest(BaseModel):
    """Body for an in-app run: collected form values + worker count."""

    values: dict[str, Any]
    jobs: int = 1


def create_app() -> FastAPI:
    """Build the *grainsmith studio* FastAPI application."""
    app = FastAPI(title="grainsmith studio", version=grainsmith.__version__)

    @app.exception_handler(ConfigError)
    async def _config_error(_request: Request, exc: ConfigError) -> JSONResponse:
        # The engine's own validation message — identical to the CLI's.
        return JSONResponse(
            status_code=422, content={"error": "ConfigError", "message": str(exc)}
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": grainsmith.__version__}

    @app.get("/api/schema")
    def get_schema() -> dict[str, Any]:
        """The schema-derived form spec — the React form is built from this."""
        return schema_json.schema_to_json()

    @app.get("/api/presets")
    def list_presets() -> dict[str, list[str]]:
        return {"presets": sorted(discover_presets())}

    @app.get("/api/presets/{name}")
    def get_preset(name: str) -> dict[str, Any]:
        try:
            config, values = load_preset(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"name": name, "yaml": config_to_yaml(config), "values": values}

    @app.post("/api/validate")
    def validate(req: ValuesRequest) -> dict[str, Any]:
        # to_yaml validates through the engine resolver; a failure raises
        # ConfigError -> 422 via the handler above.
        return {"ok": True, "yaml": to_yaml(req.values)}

    @app.post("/api/yaml/from-values")
    def yaml_from_values(req: ValuesRequest) -> dict[str, str]:
        return {"yaml": to_yaml(req.values)}

    @app.post("/api/yaml/to-values")
    def yaml_to_values(req: YamlRequest) -> dict[str, Any]:
        config, values = load_yaml(req.yaml)
        return {"values": values, "yaml": config_to_yaml(config)}

    @app.get("/api/outputs")
    def get_outputs(
        directory: str = Query(
            ..., alias="dir", description="A finished run's output directory"
        ),
    ) -> dict[str, Any]:
        outdir = Path(directory).expanduser()
        if not outdir.is_dir():
            raise HTTPException(status_code=404, detail=f"Not a directory: {outdir}")
        try:
            return outputs_json.load_outputs_json(outdir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Could not read outputs: {exc}"
            ) from exc

    @app.get("/api/pick-dir")
    def pick_dir() -> dict[str, str]:
        """Open a native folder dialog on the (local) server machine and return
        the chosen absolute path, or ``''`` when cancelled / no GUI is
        available — the viewer then keeps the manually typed path."""
        return {"dir": _native_pick_directory()}

    @app.get("/api/run/notice")
    def run_notice() -> dict[str, str]:
        """The pre-run memory caution (shown before an in-app run)."""
        return {"notice": RUN_MEMORY_NOTICE}

    @app.post("/api/run")
    def run(req: RunRequest) -> StreamingResponse:
        # Validate first: a ConfigError here surfaces as 422 (handler above)
        # before any stream is opened.
        config = config_from_values(req.values)
        # One in-app run at a time; the pipeline's logging setup is global.
        if not RUN_LOCK.acquire(blocking=False):
            raise HTTPException(
                status_code=409, detail="A run is already in progress."
            )
        return StreamingResponse(
            run_events(config, req.jobs, RUN_LOCK),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    _mount_static(app)
    return app


def _mount_static(app: FastAPI) -> None:
    """Serve the built React bundle at ``/`` when present.

    In the early phases the bundle does not exist yet; a friendly root route
    explains how to build it instead of returning a bare 404, while the JSON
    API under ``/api`` is already live.
    """
    if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
        return

    @app.get("/")
    def _no_bundle() -> dict[str, str]:
        return {
            "status": "no-frontend-bundle",
            "message": (
                "The React frontend has not been built yet. Build the Vite bundle "
                "so it is served here; the JSON API under /api is already live."
            ),
        }
