"""Static import extraction for the scientific architecture documentation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools.gen_module_graph as generator
    return generator


def test_relative_lazy_and_module_imports(tmp_path):
    generator = _generator()
    source = tmp_path / "grainsmith"
    files = {
        "__init__.py": "from . import constants\n",
        "constants.py": "EPS = 1e-10\n",
        "pipeline.py": "from grainsmith.config import schema\n"
                       "def run():\n    from .orientation import odf\n",
        "config/__init__.py": "",
        "config/schema.py": "from grainsmith import constants\n",
        "orientation/__init__.py": "",
        "orientation/odf.py": "from ..constants import EPS\n",
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    edges = generator.module_edges(source)
    assert edges == {
        ("grainsmith", "grainsmith.constants"),
        ("grainsmith.pipeline", "grainsmith.config.schema"),
        ("grainsmith.pipeline", "grainsmith.orientation.odf"),
        ("grainsmith.config.schema", "grainsmith.constants"),
        ("grainsmith.orientation.odf", "grainsmith.constants"),
    }
    assert generator.dependency_counts(edges) == {
        ("orchestration", "config"): 1,
        ("orchestration", "orientation"): 1,
        ("config", "foundation"): 1,
        ("orientation", "foundation"): 1,
    }


def test_optional_ui_and_same_layer_imports_are_not_counted():
    edges = {
        ("grainsmith.ui.server", "grainsmith.pipeline"),
        ("grainsmith.orientation.mdf", "grainsmith.orientation.odf"),
        ("grainsmith.qa", "grainsmith.orientation.odf"),
    }
    assert _generator().dependency_counts(edges) == {("orchestration", "orientation"): 1}