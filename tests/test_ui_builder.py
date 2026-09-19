"""Builder in-app run seam pin.

A config built through the UI's validator must run end-to-end via the *same*
``pipeline.run`` the page's "Run" button calls — proving the in-app run is not a
divergent code path.
"""
from __future__ import annotations

from grainsmith.pipeline import run
from grainsmith.ui.presets import discover_presets, examples_dir
from grainsmith.ui.yaml_builder import config_from_values


def _minimal_raw(outdir) -> dict:
    return {
        "meta": {"title": "ui-run", "verbose": 0},
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [40.0, 40.0, 40.0], "periodic": [True, True, True]},
        "grains": {"number": 4},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": str(outdir)},
    }


def test_ui_built_config_runs(tmp_path):
    cfg = config_from_values(_minimal_raw(tmp_path / "out"))
    result = run(cfg)
    assert result.gates.results()
    assert result.outdir.exists()
    assert len(result.atoms) > 0


def test_discover_presets_stems_are_unique_tree_wide():
    """discover_presets() keys its mapping by file stem, so two ``*.yaml``
    files sharing a stem in different examples/ subfolders would silently
    collapse to one entry (dict overwrite) and drop a preset from the UI's
    list without any error. Pin stem-uniqueness across the whole tree by
    comparing the preset count against a plain recursive file count."""
    presets = discover_presets()
    yaml_files = list(examples_dir().rglob("*.yaml"))
    assert len(presets) == len(yaml_files), (
        f"{len(yaml_files)} examples/**/*.yaml files but only "
        f"{len(presets)} discovered presets — a duplicate file stem is "
        "silently dropping a preset"
    )
