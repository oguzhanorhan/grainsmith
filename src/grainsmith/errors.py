"""Custom exception hierarchy for grainsmith.

Hierarchy
---------
GrainsmithError (base)
├── ConfigError        — invalid configuration
├── CrystalError       — space-group or lattice build failure
├── TessellationError  — Voronoi/GRF construction failure
├── FillError          — atom-fill failure
├── QAGateError        — QA gate assertion failure
└── ManifestError      — MANIFEST.txt missing or malformed

No bare ``except:`` clauses are used anywhere in this codebase; every
catch site names at least one concrete exception type.
"""


class GrainsmithError(Exception):
    """Base class for all grainsmith errors.

    All grainsmith-specific exceptions inherit from this class, allowing
    callers to catch the entire hierarchy with a single
    ``except GrainsmithError`` block while still being able to distinguish
    individual failure modes via subclasses.
    """


class ConfigError(GrainsmithError):
    """Raised when the user-supplied configuration is invalid.

    Examples: unknown YAML keys (extra="forbid" violation), cross-field
    constraint violations (e.g. vacuum > 0 on a fully-periodic box),
    over- or under-specification of lattice parameters for the requested
    crystal family.  The message always names the offending field(s) and
    states the correction.
    """


class CrystalError(GrainsmithError):
    """Raised when space-group or crystal-structure construction fails.

    Examples: spglib round-trip mismatch (gate G2), unsupported
    space-group number, inconsistent Wyckoff letter supplied by the user,
    singular cell matrix (v² ≤ 0 in §6.1 formula).
    """


class TessellationError(GrainsmithError):
    """Raised when the Voronoi or GRF tessellation cannot be built.

    Examples: RSA seeding exceeds the maximum attempt budget
    (RNG_MAX_ATTEMPTS_FACTOR × N), the warp-field bijectivity guard
    fails (max‖∇u‖ ≥ WARP_GRAD_MAX), or a grain becomes empty or
    disconnected after warping (gates G5 / G6).
    """


class FillError(GrainsmithError):
    """Raised when atom-filling a grain fails.

    Examples: the membership filter returns zero atoms for a non-empty
    grain (likely a bounding-radius underestimate), a singular rotation
    matrix, or a fill-chunk exceeds the FILL_CHUNK memory mandate.
    """


class QAGateError(GrainsmithError):
    """Raised when a mandatory (require-d) QA gate reports failure.

    The gate identifier (e.g. ``"G3"``), the measured value, and the
    threshold are embedded in the message so the user can diagnose and
    fix the input without reading source code.
    """


class ManifestError(GrainsmithError):
    """Raised when a run's ``MANIFEST.txt`` cannot be read as a manifest.

    Examples: the file is absent from the output directory, its first line is
    not a grainsmith provenance header, a digest line does not carry exactly
    ``sha256 size path``, or the full-length ``config_sha256`` /
    ``provenance_sha256`` comment fields are missing. Distinguished from a
    *verification failure* (a digest that does not match), which is reported
    through the ``grainsmith verify`` report and its exit code, not raised.
    """
