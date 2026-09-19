"""Tests for grainsmith.provenance: the version-bound provenance digest (R1),
SOURCE_DATE_EPOCH timestamp freeze, and the resolved_config.yaml round-trip
property that makes ``grainsmith verify`` possible from shipped files alone."""
from __future__ import annotations

import re

import pytest

from grainsmith.config.resolve import config_sha12, config_sha256, dump_resolved, resolve_config
from grainsmith.errors import ConfigError
from grainsmith.pipeline import run
from grainsmith.provenance import (
    ENVIRONMENT_KEYS,
    _blas_record,
    environment_record,
    provenance_payload,
    provenance_sha256,
    run_timestamp_iso,
)

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _raw(**overrides) -> dict:
    raw = {
        "seed": {"mode": "fixed", "value": 1},
        "box": {"lengths": [12.0, 12.0, 12.0]},
        "grains": {"number": 2},
        "crystal": {
            "space_group": {"number": 225},
            "lattice": {"a": 3.615},
            "wyckoff_sites": [{"element": "Cu", "coords": [0.0, 0.0, 0.0]}],
        },
        "output": {"directory": "unused"},
    }
    raw.update(overrides)
    return raw


def _config(outdir):
    return resolve_config(_raw(output={"directory": str(outdir)}))


# ---------------------------------------------------------------------------
# T1: the payload is pinned, byte-exact
# ---------------------------------------------------------------------------


def test_payload_is_pinned():
    payload = provenance_payload("1.1.0", "0" * 64)
    assert payload == (
        "grainsmith-provenance/v1\n"
        "grainsmith_version=1.1.0\n"
        "config_sha256=" + "0" * 64 + "\n"
    )


def test_digest_is_pinned():
    # Pinned: changing either constant means the provenance payload format
    # changed, which is a BREAKING change to every output's version binding.
    assert provenance_sha256("1.1.0", "0" * 64) == (
        "f954a9dace02d020c06316110df6085e4506fcf47e1e4f143a742e34534ec4b3")
    assert provenance_sha256("1.2.0", "0" * 64) == (
        "27a2ae6e8e13f4a19cb4a5b0da479c15739a8a737975398cb4b900063db2953f")


# ---------------------------------------------------------------------------
# T2: the version is INSIDE the payload — the R1 property
# ---------------------------------------------------------------------------


def test_version_is_bound_into_the_digest():
    d = "a" * 64
    prov_a = provenance_sha256("1.1.0", d)
    prov_b = provenance_sha256("1.2.0", d)
    assert prov_a != prov_b, "two versions of the same config must bind differently"
    # ... while config_sha256 itself carries no version at all: it is
    # computed independently of provenance_sha256 (grainsmith.config.resolve
    # never sees a version string), so THAT half of the pair is unaffected —
    # only the version-bound digest changes.
    cfg1 = resolve_config(_raw())
    cfg2 = resolve_config(_raw())
    assert config_sha256(cfg1) == config_sha256(cfg2)


# ---------------------------------------------------------------------------
# T3: config_sha256 is a pure function of the resolved config
# ---------------------------------------------------------------------------


def test_config_sha256_pure_function_of_config():
    cfg1 = resolve_config(_raw())
    cfg2 = resolve_config(_raw())
    assert config_sha256(cfg1) == config_sha256(cfg2)

    cfg3 = resolve_config(_raw(grains={"number": 3}))
    assert config_sha256(cfg1) != config_sha256(cfg3)


# ---------------------------------------------------------------------------
# T4: config_sha12 is exactly the truncation
# ---------------------------------------------------------------------------


def test_config_sha12_is_the_truncation():
    cfg = resolve_config(_raw())
    assert config_sha12(cfg) == config_sha256(cfg)[:12]
    assert len(config_sha12(cfg)) == 12


# ---------------------------------------------------------------------------
# T5: SOURCE_DATE_EPOCH freezes the timestamp
# ---------------------------------------------------------------------------


def test_source_date_epoch_freezes_timestamp(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    assert run_timestamp_iso() == "2023-11-14T22:13:20Z"


def test_source_date_epoch_unset_uses_wall_clock(monkeypatch):
    import time

    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    before = time.time()
    ts = run_timestamp_iso()
    after = time.time()
    assert _ISO_RE.match(ts)
    from datetime import datetime, timezone
    parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    assert before - 5 <= parsed.timestamp() <= after + 5


# ---------------------------------------------------------------------------
# T6: a bad SOURCE_DATE_EPOCH raises, never silently falls back
# ---------------------------------------------------------------------------


def test_source_date_epoch_non_integer_raises(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-a-number")
    with pytest.raises(ConfigError, match="SOURCE_DATE_EPOCH"):
        run_timestamp_iso()


def test_source_date_epoch_negative_raises(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "-1")
    with pytest.raises(ConfigError, match="SOURCE_DATE_EPOCH"):
        run_timestamp_iso()


def test_source_date_epoch_bad_value_raises_from_run_before_numpy_sees_it(
        tmp_path, monkeypatch):
    """Regression: numpy's f2py reads SOURCE_DATE_EPOCH too, during scipy's
    lazy import inside run() -- so grainsmith's own validation must happen
    BEFORE that import chain, or a typo surfaces as a bare numpy ValueError
    (a 40-frame traceback) instead of this package's clear ConfigError."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "oops")
    with pytest.raises(ConfigError) as exc_info:
        run(_config(tmp_path / "out"))
    assert "SOURCE_DATE_EPOCH" in str(exc_info.value)


# ---------------------------------------------------------------------------
# T7: Provenance.line() format
# ---------------------------------------------------------------------------


def test_provenance_line_format():
    from grainsmith.provenance import make_provenance

    prov = make_provenance("1.1.0", "2026-09-01T12:00:00Z", 555666,
                           "fd1c4ecb04ac" + "0" * 52, title="t")
    expected_provenance_sha = provenance_sha256(
        "1.1.0", "fd1c4ecb04ac" + "0" * 52)
    assert prov.line() == (
        "grainsmith 1.1.0 | 2026-09-01T12:00:00Z | seed=555666 | "
        "config sha256=fd1c4ecb04ac | "
        f"provenance sha256={expected_provenance_sha[:12]}"
    )
    assert prov.provenance_sha256 == expected_provenance_sha


# ---------------------------------------------------------------------------
# T8: resolved_config.yaml round-trip property — the single most important
# regression guard for `grainsmith verify` (§5.2-C4)
# ---------------------------------------------------------------------------


def test_resolved_config_round_trip_property(tmp_path):
    import hashlib

    cfg = resolve_config(_raw())
    path = tmp_path / "resolved_config.yaml"
    dump_resolved(cfg, path, timestamp_iso="2026-09-01T12:00:00Z", seed=1)

    raw = path.read_bytes()
    lines = raw.split(b"\n")
    i = 0
    while i < len(lines) and lines[i].startswith(b"#"):
        i += 1
    body = b"\n".join(lines[i:])
    assert hashlib.sha256(body).hexdigest() == config_sha256(cfg)

    # Exactly two leading "#" lines, both starting with "#" -- a third
    # header line (or a stray blank line) would silently break the
    # round-trip that `grainsmith verify` relies on.
    assert i == 2
    assert lines[0].startswith(b"#") and lines[1].startswith(b"#")


# ---------------------------------------------------------------------------
# E1-E5: environment_record() (§R3) -- the BLAS/platform/numba/jobs record
# behind a run's floating-point bytes.
# ---------------------------------------------------------------------------


def test_environment_record_every_key_present_and_stringly_typed():
    """E1: exactly ENVIRONMENT_KEYS, every value a non-empty str."""
    env = environment_record()
    assert set(env) == set(ENVIRONMENT_KEYS)
    for key, value in env.items():
        assert isinstance(value, str), f"{key} is not a str: {value!r}"
        assert value != "", f"{key} is an empty string"


def test_environment_record_jobs_round_trip():
    """E2: jobs_requested/jobs_effective pass through as strings; absent ->
    "unknown" (the value the caller passed to run()/--jobs, distinct from
    the post-expansion worker count -- see pipeline.py's `--jobs 0` comment)."""
    env = environment_record(0, 12)
    assert env["jobs_requested"] == "0"
    assert env["jobs_effective"] == "12"

    env_default = environment_record()
    assert env_default["jobs_requested"] == "unknown"
    assert env_default["jobs_effective"] == "unknown"


def test_blas_record_numpy_124_fallback_path(monkeypatch):
    """E3: numpy 1.24 has no `mode=` kwarg on show_config() -- a zero-arg
    callable reproduces that TypeError on this (numpy >= 2) environment,
    which is the only way to exercise the legacy `numpy.__config__` branch
    here without actually installing numpy 1.24."""
    import numpy as np

    monkeypatch.setattr(np, "show_config", lambda: None)
    result = _blas_record()
    assert set(result) == {"blas_name", "blas_version", "blas_detection",
                           "blas_openblas_config"}


def test_blas_record_total_probe_failure_never_raises(monkeypatch):
    """E4: guards the "provenance capture must not fail a successful run"
    contract -- an arbitrary exception from show_config() must still return
    an all-"unknown" record, never propagate."""
    import numpy as np

    def _boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(np, "show_config", _boom)
    result = _blas_record()
    assert result == {
        "blas_name": "unknown", "blas_version": "unknown",
        "blas_detection": "unknown", "blas_openblas_config": "unknown",
    }


def test_numba_record_matches_this_environment():
    """E5a: the numba record must agree with whether numba is ACTUALLY
    importable here, in either direction.

    numba is an OPTIONAL accelerator (the ``owns()`` kernels), so both
    environments are legitimate and this test has to hold in both: it used
    to hard-code ``== "absent"``, which silently encoded "the machine that
    wrote this test had no numba" and failed for every contributor who did
    have it installed. Deriving the expectation from a real import keeps
    the assertion just as strict without pinning the environment.
    """
    env = environment_record()
    try:
        import numba
    except ImportError:
        assert env["numba"] == "absent"
        return
    # Installed: the record must name the real version and one of the
    # three dispatch states _numba_record() can report (provenance.py).
    assert env["numba"].startswith(f"{numba.__version__} (")
    assert env["numba"].endswith(")")
    state = env["numba"][len(numba.__version__) + 2:-1]
    assert state in {"active", "disabled: GRAINSMITH_NO_NUMBA",
                     "dispatch state unknown"}


def test_numba_record_disabled_by_env_with_injected_fake_module(monkeypatch):
    """E5b: cover the "(active)" / "(disabled: GRAINSMITH_NO_NUMBA)"
    branches deterministically, by injecting a fake numba module into
    sys.modules and monkeypatching weighted._USE_NUMBA directly.

    numba is an OPTIONAL accelerator, so it may or may not be installed
    where this suite runs; faking it is what lets BOTH branches be
    exercised either way, instead of depending on the host environment
    (see test_numba_record_matches_this_environment for the real one)."""
    import sys
    import types

    from grainsmith.tessellation import weighted

    fake_numba = types.ModuleType("numba")
    fake_numba.__version__ = "0.60.0"
    monkeypatch.setitem(sys.modules, "numba", fake_numba)
    monkeypatch.setattr(weighted, "_USE_NUMBA", False)

    env = environment_record()
    assert env["numba"] == "0.60.0 (disabled: GRAINSMITH_NO_NUMBA)"
