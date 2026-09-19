"""Smoke pins for the UI presentation layer ("grainsmith studio").

Skipped wholesale when the optional ``[ui]`` extra is absent (fastapi not
installed); in CI the extra is installed, so these run there.  They check only
that the thin presentation seam imports and exposes its surface — the real
logic is covered by the pure-module tests (formspec / yaml_builder / analysis /
schema_json / outputs_json) and the endpoint behaviour by ``test_ui_server``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


def test_launch_seam():
    """The launcher entry points are importable from the package root."""
    from grainsmith.ui import APP_TITLE, launch

    assert callable(launch)
    assert APP_TITLE


def test_create_app_builds_fastapi_app():
    """The server factory returns a configured FastAPI app without running it."""
    from fastapi import FastAPI

    from grainsmith.ui.server import create_app

    app = create_app()
    assert isinstance(app, FastAPI)
    routes = {getattr(r, "path", None) for r in app.routes}
    # the schema endpoint (which re-pins the no-field-drop guarantee) is mounted
    assert "/api/schema" in routes
    assert "/api/health" in routes
