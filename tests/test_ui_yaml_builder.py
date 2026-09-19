"""yaml_builder pins.

The builder routes every value through the engine's own resolver, so: a valid
mapping yields a RunConfig; a YAML round-trip is stable; every shipped preset
round-trips; and bad input (unknown key / out-of-range) is rejected with the
engine's ConfigError — the UI can never emit a config the CLI would reject.
"""
from __future__ import annotations

import pytest

from grainsmith.config.schema import RunConfig
from grainsmith.errors import ConfigError
from grainsmith.ui.presets import discover_presets
from grainsmith.ui.yaml_builder import config_from_values, config_to_yaml, load_yaml


def _minimal_raw(outdir) -> dict:
    """The §10 minimal FCC Cu config (mirrors tests/test_end_to_end.py)."""
    return {
        "meta": {"title": "ui", "verbose": 0},
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


def test_config_from_values_minimal(tmp_path):
    cfg = config_from_values(_minimal_raw(tmp_path / "out"))
    assert isinstance(cfg, RunConfig)


def test_yaml_round_trip_is_stable(tmp_path):
    cfg = config_from_values(_minimal_raw(tmp_path / "out"))
    cfg2, _ = load_yaml(config_to_yaml(cfg))
    assert cfg2.model_dump() == cfg.model_dump()


@pytest.mark.parametrize("path", sorted(discover_presets().values()))
def test_every_preset_round_trips(path):
    cfg, _ = load_yaml(path)
    cfg2, _ = load_yaml(config_to_yaml(cfg))
    assert cfg2.model_dump() == cfg.model_dump()


def test_unknown_key_rejected(tmp_path):
    raw = _minimal_raw(tmp_path / "out")
    raw["meta"]["not_a_field"] = 1
    with pytest.raises(ConfigError):
        config_from_values(raw)


def test_out_of_range_rejected(tmp_path):
    raw = _minimal_raw(tmp_path / "out")
    raw["meta"]["verbose"] = 99  # schema enforces le=3
    with pytest.raises(ConfigError):
        config_from_values(raw)


def test_load_yaml_str_is_parsed_as_text_never_read_as_file(tmp_path):
    """A ``str`` argument must ALWAYS be parsed as YAML text, never used as a
    filename.  The /api/yaml/to-values endpoint forwards raw client text here,
    so reading a file whose name the text happens to match (e.g. '/etc/hostname')
    would be a local file-disclosure vector."""
    secret = tmp_path / "secret.yaml"
    secret.write_text("seed: 12345\n")          # a valid YAML mapping ON DISK
    # The path STRING is a scalar when parsed as YAML text -> not a mapping.
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(str(secret))


def test_load_yaml_path_still_reads_from_disk(tmp_path):
    """The explicit pathlib.Path branch (used by presets) still reads the file."""
    from pathlib import Path
    f = tmp_path / "cfg.yaml"
    f.write_text("not_a_real_key: 1\n")
    # Read from disk -> mapping -> rejected by the resolver for an unknown key,
    # proving the file WAS read (a different error than the 'mapping' one above).
    with pytest.raises(ConfigError) as exc:
        load_yaml(Path(f))
    assert "mapping" not in str(exc.value)


def test_load_yaml_str_gets_no_base_dir_fallback(examples_dir, tmp_path, monkeypatch):
    """A ``str`` payload (e.g. the POST /api/yaml/to-values body) has no
    filesystem location of its own to anchor a config-directory fallback
    against, unlike the ``Path`` branch (a shipped preset, see
    test_load_yaml_path_still_reads_from_disk above) which DOES get one
    -- see grainsmith.config.resolve.resolve_config's base_dir. A
    crystal.cif.file relative path that exists only relative to
    examples_dir (never relative to this test's CWD) must therefore still
    raise ConfigError when passed as raw text."""
    # sanity: the path really exists, relative to examples_dir -- a
    # Path-based load of an examples/*.yaml preset WOULD resolve it via
    # base_dir; this test proves the str branch does not get that fallback.
    assert (examples_dir / "assets" / "TiNi.cif").exists()

    monkeypatch.chdir(tmp_path)   # nothing named 'assets' here
    raw_yaml = (
        "seed: {mode: fixed, value: 1}\n"
        "box: {lengths: [40.0, 40.0, 40.0], periodic: [true, true, true]}\n"
        "grains: {number: 4}\n"
        "crystal:\n"
        "  cif:\n"
        "    file: assets/TiNi.cif\n"
    )
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(raw_yaml)


def test_load_preset_str_never_reads_arbitrary_file(tmp_path, monkeypatch):
    """A bare non-preset name must raise KeyError, not read a same-named file
    from the process working directory (GET /api/presets/{name} forwards the raw
    URL segment here)."""
    from grainsmith.ui.presets import load_preset
    bait = tmp_path / "notapreset.yaml"
    bait.write_text("seed: 1\n")
    monkeypatch.chdir(tmp_path)                  # the bait file is now cwd-relative
    with pytest.raises(KeyError):
        load_preset("notapreset.yaml")
