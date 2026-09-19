"""Version-bound provenance digest + reproducible-build timestamp freeze.

Two digests travel with every run (§8): ``config.resolve.config_sha256``
identifies the *input* (the resolved config alone — two grainsmith versions
run on the same config produce the same value), and :func:`provenance_sha256`
identifies *input + producer* by hashing a payload that contains the
grainsmith version string. Because the version is inside the hashed bytes, an
output's version claim is cryptographically bound to it: editing the recorded
version anywhere invalidates the digest, which is exactly what ``grainsmith
verify`` checks (check C5).

This module is intentionally light: pure stdlib, no dependency on
``grainsmith.io`` at module scope (``io.common`` builds a ``Provenance`` via
:func:`make_provenance`, so a module-scope import the other way would create
a cycle), and no dependency on ``grainsmith.pipeline`` (``grainsmith.verify``
must be importable — and fast — in a minimal environment with no numpy/scipy/
spglib installed).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from grainsmith.errors import ConfigError

if TYPE_CHECKING:
    from grainsmith.io.common import Provenance

PROVENANCE_PAYLOAD_SCHEMA = "grainsmith-provenance/v1"
"""Schema tag for :func:`provenance_payload` — the first field recorded in
every ``provenance_payload_schema`` output (summary.csv, MANIFEST.txt,
microstructure.json), so a future payload-format change is self-describing."""


def provenance_payload(version: str, config_sha256: str) -> str:
    """The byte-exact payload behind ``provenance_sha256``.

    Deliberately tiny and fully reconstructible from shipped files: the schema
    tag, the producing grainsmith version, and the full config digest. Because
    the version string is INSIDE the hashed bytes, an output cannot claim
    "produced with v1.1" unless it really was — editing the recorded version
    string anywhere invalidates the digest (``grainsmith verify`` check C5).

    NOT included, on purpose:
      * the timestamp and the seed — they are already in the header line, and
        keeping them out makes the digest recomputable from
        ``resolved_config.yaml`` + the recorded version alone;
      * anything environment-dependent — the digest must be identical on every
        machine that runs the same version on the same config.
    """
    return (f"{PROVENANCE_PAYLOAD_SCHEMA}\n"
            f"grainsmith_version={version}\n"
            f"config_sha256={config_sha256}\n")


def provenance_sha256(version: str, config_sha256: str) -> str:
    """Full SHA-256 (64 hex) of :func:`provenance_payload`."""
    return hashlib.sha256(
        provenance_payload(version, config_sha256).encode("utf-8")
    ).hexdigest()


def make_provenance(version: str, timestamp_iso: str, seed: int,
                    config_sha256: str, title: str = "") -> Provenance:
    """Build the single per-run Provenance record, deriving the version-bound
    digest so no call site can forget it (there is exactly one construction
    path for the two digests)."""
    from grainsmith.io.common import Provenance
    return Provenance(
        version=version, timestamp_iso=timestamp_iso, seed=seed,
        config_sha256=config_sha256,
        provenance_sha256=provenance_sha256(version, config_sha256),
        title=title,
    )


def source_date_epoch() -> int | None:
    """The reproducible-builds ``SOURCE_DATE_EPOCH`` override, or None.

    Standard semantics (https://reproducible-builds.org/docs/source-date-epoch/):
    a decimal integer count of seconds since the Unix epoch, interpreted as UTC.
    Unset or empty -> None (the run stamps the real wall clock, as before).

    An unparseable / negative value raises rather than silently falling back to
    the wall clock: a caller who set the variable is explicitly asking for a
    frozen clock, and quietly ignoring a typo would produce an output that
    *looks* reproducible and is not (house rule: no silent fallbacks).
    """
    raw = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"SOURCE_DATE_EPOCH must be an integer number of seconds since "
            f"the Unix epoch (UTC), got {raw!r}."
        ) from exc
    if value < 0:
        raise ConfigError(
            f"SOURCE_DATE_EPOCH must be >= 0, got {value}.")
    return value


def run_timestamp_iso() -> str:
    """The single per-run UTC timestamp stamped into every provenance header.

    ``SOURCE_DATE_EPOCH`` freezes it (making MANIFEST.txt and every header
    byte-reproducible); unset, it is ``datetime.now(timezone.utc)`` exactly as
    before. Format is unchanged: ``%Y-%m-%dT%H:%M:%SZ``.
    """
    epoch = source_date_epoch()
    when = (datetime.fromtimestamp(epoch, tz=timezone.utc) if epoch is not None
            else datetime.now(tz=timezone.utc))
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Environment provenance (R3): the execution environment behind a run's
# floating-point bytes -- see environment_record() below.
# ---------------------------------------------------------------------------

ENVIRONMENT_KEYS: tuple[str, ...] = (
    "python_implementation", "python_version",
    "os", "os_release", "platform", "machine", "processor", "libc",
    "blas_name", "blas_version", "blas_detection", "blas_openblas_config",
    "numba", "jobs_requested", "jobs_effective",
)
"""Fixed key order for the environment record.

Pinned as a tuple, not derived from a dict, because these keys become
``summary.csv`` rows and a reordering would change the file bytes for no
scientific reason (the byte-identical re-run contract, §10)."""


def _blas_record_legacy(unknown: dict[str, str]) -> dict[str, str]:
    """numpy 1.24 fallback: the pre-meson ``numpy.__config__`` info dicts."""
    try:
        from numpy import __config__ as npcfg
    except ImportError:                              # pragma: no cover
        return dict(unknown)
    for attr in ("blas_ilp64_opt_info", "blas_opt_info",
                 "openblas_info", "blas_info"):
        info = getattr(npcfg, attr, None)
        if isinstance(info, dict) and info.get("libraries"):
            out = dict(unknown)
            out["blas_name"] = ",".join(str(x) for x in info["libraries"])
            out["blas_detection"] = f"numpy.__config__.{attr}"
            return out
    return dict(unknown)


def _blas_record() -> dict[str, str]:
    """BLAS vendor/version behind numpy's ``@`` — the dominant cross-machine
    byte-identity factor (measured: 8.6-28.4 % of candidate coordinates differ
    in the last bit between two BLAS routes on the same machine).

    numpy >= 1.25 exposes a structured ``show_config(mode="dicts")``; numpy 1.24
    (this project's declared floor, pyproject.toml) has no ``mode`` parameter at
    all and raises TypeError, so that branch reads the pre-meson
    ``numpy.__config__`` info dicts instead. Both paths are best-effort: an
    unknown BLAS is recorded as "unknown", never raised, because provenance
    capture must not fail a run whose science already succeeded.
    """
    unknown = {"blas_name": "unknown", "blas_version": "unknown",
               "blas_detection": "unknown", "blas_openblas_config": "unknown"}
    import numpy as np
    try:
        cfg = np.show_config(mode="dicts")          # numpy >= 1.25
    except TypeError:                                # numpy 1.24: no mode kwarg
        return _blas_record_legacy(unknown)
    except Exception:                                # any probe failure at all
        # Catching bare Exception is normally against house style; here it is
        # correct (and must keep this comment): a provenance probe may never
        # abort a run whose science already succeeded. The `except TypeError`
        # branch above is deliberately checked FIRST so the intentional
        # numpy-1.24 case is never swallowed by this catch-all.
        return dict(unknown)
    blas = ((cfg or {}).get("Build Dependencies") or {}).get("blas") or {}
    if not isinstance(blas, dict):
        return dict(unknown)
    return {
        "blas_name": str(blas.get("name") or "unknown"),
        "blas_version": str(blas.get("version") or "unknown"),
        "blas_detection": str(blas.get("detection method") or "unknown"),
        # OpenBLAS wheels put the full build string here (target CPU kernel,
        # thread model, ILP64 flag) — the highest-value single field for
        # explaining a cross-machine last-bit difference. Generic/conda BLAS
        # reports the literal "unknown".
        "blas_openblas_config": str(
            blas.get("openblas configuration") or "unknown"),
    }


def _numba_record() -> str:
    """Whether the optional numba owns()-kernel path was ACTIVE this run.

    ``tessellation/weighted.py:130`` computes ``_USE_NUMBA`` once at import
    time from availability AND ``GRAINSMITH_NO_NUMBA``; that flag is what the
    engine actually dispatches on (weighted.py:421), so it is the single
    source of truth here. (``tessellation/_local_owns.py:37`` has its own
    ``_USE_NUMBA`` that tracks availability only, with no env gate — both
    kernels are bit-identical to the numpy reference by the repo's own A/B
    tests, so one flag is enough for provenance.)
    """
    try:
        import numba
    except ImportError:
        return "absent"
    try:
        from grainsmith.tessellation import weighted
    except ImportError:                              # pragma: no cover
        return f"{numba.__version__} (dispatch state unknown)"
    return (f"{numba.__version__} (active)" if weighted._USE_NUMBA
            else f"{numba.__version__} (disabled: GRAINSMITH_NO_NUMBA)")


def environment_record(jobs_requested: int | None = None,
                       jobs_effective: int | None = None) -> dict[str, str]:
    """The execution environment behind a run's floating-point bytes.

    Recorded because the same seed + same grainsmith version does NOT imply the
    same output bytes across machines: the atom-position hot path
    ``(A @ F_flat.T).T`` (atoms/fill.py) dispatches to BLAS ``dgemm``, and a
    different BLAS vendor/build or CPU ISA changes the last bit of a measurable
    fraction of coordinates. None of that changes the physics (grain ownership
    has a ~9-orders-of-magnitude margin, measured), but it does change the
    digests in MANIFEST.txt — so a reader comparing digests across machines
    needs to see WHICH environment produced them.

    Every value is a plain ``str``; never raises (an unavailable probe records
    ``"unknown"``), because provenance capture must not be able to fail a run
    that has already produced correct science.
    """
    import platform as _platform

    def _or_unknown(value: str) -> str:
        return value if value else "unknown"

    try:
        libc = " ".join(_platform.libc_ver()).strip()
    except Exception:                                # pragma: no cover
        libc = ""

    blas = _blas_record()
    # Built in ENVIRONMENT_KEYS order (not assembled via dict.update() after
    # the fact): this dict is what write_microstructure_json embeds verbatim
    # as the JSON "environment" object, and although the pinned ORDER only
    # matters for summary.csv's row sequence (§ENVIRONMENT_KEYS docstring),
    # matching it here too means both files read the same to a human diffing
    # them side by side.
    record: dict[str, str] = {
        "python_implementation": _platform.python_implementation(),
        "python_version": _platform.python_version(),
        "os": _or_unknown(_platform.system()),
        "os_release": _or_unknown(_platform.release()),
        "platform": _or_unknown(_platform.platform()),
        "machine": _or_unknown(_platform.machine()),
        "processor": _or_unknown(_platform.processor()),
        "libc": _or_unknown(libc),
        "blas_name": blas["blas_name"],
        "blas_version": blas["blas_version"],
        "blas_detection": blas["blas_detection"],
        "blas_openblas_config": blas["blas_openblas_config"],
        "numba": _numba_record(),
        "jobs_requested": ("unknown" if jobs_requested is None
                           else str(jobs_requested)),
        "jobs_effective": ("unknown" if jobs_effective is None
                           else str(jobs_effective)),
    }
    return record
