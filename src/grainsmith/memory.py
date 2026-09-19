"""§13 memory-mandate support: RAM-aware budget resolution + an optional
best-effort total-RSS monitor.

The §13 memory mandate (``constants.MEMORY_HARD_LIMIT_BYTES``, overridable
per run via ``runtime.memory_limit_gb``) guards *individual* large
allocations. Two independent things live in this module:

1. :func:`detect_physical_ram_bytes` / :func:`resolve_memory_budget` /
   :func:`log_effective_budget` / :func:`format_bytes_adaptive` /
   :func:`build_memory_guard_message` -- the RAM-CLAMPED effective
   per-allocation budget (the configured value can never exceed detected
   physical RAM) and the shared error-message builder every §13 guard site
   (atoms/fill.py, tessellation/voxel.py, tessellation/voxel_import.py) now
   calls. ``pipeline.run`` resolves this ONCE per run, near the top,
   before any of its tessellation-binding branches constructs anything,
   logs the single effective-budget INFO line via
   :func:`log_effective_budget`, and threads the resolved tuple through to
   CONSTRUCTION time for every curved tessellation backend (see
   ``pipeline._stage_tessellation`` / ``pipeline._apply_runtime_limits``
   and ``tessellation/base.py``'s ``Tessellation.memory_limit_bytes``/
   ``memory_limit_source``) -- both accept the tuple as an optional
   parameter and only fall back to resolving (and logging) it themselves
   when called standalone, outside ``pipeline.run``. The detected/clamped
    values are MACHINE-DEPENDENT. They are reported as observations in
    summary.csv and run.log, never substituted for the user's configured
    value in resolved_config.yaml or the configuration hash.
2. :class:`MemoryMonitor` -- a *soft*, best-effort total-RSS monitor,
   covering what the per-allocation guard does not: the cumulative
   resident set (RSS) of the driver process, which can climb well past
   any single allocation — measured peaking near 13.1 GB on a large run
   while every per-allocation estimate stayed under the (then lower)
   per-allocation guard in effect at the time. On an HPC batch queue that
   surfaces as a silent OOM-kill.

   It samples the process RSS on a background thread, records the peak,
    and WARNs (via logging — it never raises or changes the scientific
    structure) when a ceiling is crossed. The ceiling is the
   user-supplied ``--max-rss`` (GB) if given, else
   ``MEMORY_SOFT_WARN_FRACTION`` of physical RAM.

   It depends on the optional :mod:`psutil` package. If psutil is not
   importable the monitor is inert (:attr:`available` is ``False``) and
   the run proceeds exactly as before. RSS is inherently
    non-deterministic; snapshots are reported in summary.csv as measured
    runtime data, as well as logging and the pipeline RunResult.
"""
from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

from grainsmith.constants import (
    MEMORY_MONITOR_POLL_SECONDS,
    MEMORY_SOFT_WARN_FRACTION,
)

if TYPE_CHECKING:
    from grainsmith.config.schema import RunConfig

log = logging.getLogger(__name__)

_GB = 1e9  # decimal GB (matches how RAM and the --max-rss flag are quoted)
_MB = 1e6  # decimal MB, same convention


# ---------------------------------------------------------------------------
# Physical-RAM detection + the RAM-clamped §13 effective budget
# ---------------------------------------------------------------------------


def detect_available_ram_bytes() -> float | None:
    """Best-effort CURRENTLY AVAILABLE RAM in bytes (what a new allocation
    can actually get without swapping), or ``None`` if unknown. NEVER raises.

    Distinct from :func:`detect_physical_ram_bytes`, which reports the
    machine's TOTAL RAM. The distinction matters on an interactive
    workstation: a 31 GB laptop running an IDE and a browser typically has
    only ~11 GB available, so a budget derived from the total silently
    licenses an allocation the kernel will OOM-kill. Used by
    ``atoms.fill.fill_grains`` to size the concurrent-worker pre-flight.

    1. :mod:`psutil` ``virtual_memory().available`` -- accounts for
       reclaimable page cache, so it is the right number, not ``free``.
    2. Linux ``/proc/meminfo``'s ``MemAvailable:`` line (kB), which the
       kernel computes with the same intent.

    Deliberately has NO total-RAM fallback: returning the total here would
    defeat the purpose, so callers get ``None`` and must decide for
    themselves (``fill_grains`` then falls back to the configured
    per-allocation budget alone).

    The machine-dependent result is diagnostic metadata. It must not
    replace any configured value in the resolved input or its hash.
    """
    # 1. psutil
    try:
        import psutil

        avail = float(psutil.virtual_memory().available)
        if avail > 0:
            return avail
    except Exception:
        pass

    # 2. Linux /proc/meminfo MemAvailable
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    if kb > 0:
                        return kb * 1024.0
                    break
    except Exception:
        pass

    return None


def detect_physical_ram_bytes() -> float | None:
    """Best-effort total PHYSICAL RAM in bytes. Tried in order, each
    independently guarded so a failure/absence of one path falls through
    to the next; NEVER raises. Returns ``None`` if every path fails (e.g.
    a sandboxed or otherwise unsupported platform), so a caller can fall
    back to trusting the configured value unclamped (see
    :func:`resolve_memory_budget`).

    1. :mod:`psutil` -- an OPTIONAL extra (pyproject.toml's ``monitor``
       extra, the same one :class:`MemoryMonitor` uses), never promoted to
       a core dependency for this function alone.
    2. POSIX ``os.sysconf`` (``SC_PAGE_SIZE`` x ``SC_PHYS_PAGES``) -- most
       Unix, including Linux. ``SC_PHYS_PAGES`` is absent from
       ``os.sysconf_names`` on macOS, hence the explicit membership guard.
    3. Linux ``/proc/meminfo``'s ``MemTotal:`` line (kB).
    4. Windows: ``ctypes`` ``MEMORYSTATUSEX`` +
       ``kernel32!GlobalMemoryStatusEx``.
    5. macOS: ``sysctl -n hw.memsize`` (bytes) via subprocess.

    The machine-dependent result is diagnostic metadata. It must not
    replace any configured value in the resolved input or its hash.
    """
    # 1. psutil
    try:
        import psutil

        total = float(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass

    # 2. POSIX sysconf
    try:
        if (hasattr(os, "sysconf") and hasattr(os, "sysconf_names")
                and "SC_PAGE_SIZE" in os.sysconf_names
                and "SC_PHYS_PAGES" in os.sysconf_names):
            total = float(os.sysconf("SC_PAGE_SIZE")) * float(
                os.sysconf("SC_PHYS_PAGES"))
            if total > 0:
                return total
    except (ValueError, OSError):
        pass

    # 3. Linux /proc/meminfo
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    if kb > 0:
                        return kb * 1024.0
                    break
    except (OSError, ValueError, IndexError):
        pass

    # 4. Windows
    try:
        if sys.platform.startswith("win"):
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total = float(stat.ullTotalPhys)
                if total > 0:
                    return total
    except Exception:
        pass

    # 5. macOS
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True,
                text=True, timeout=5, check=True)
            total = float(out.stdout.strip())
            if total > 0:
                return total
    except Exception:
        pass

    return None


def resolve_memory_budget(
    config: RunConfig,
) -> tuple[float, str, float | None]:
    """Resolve THIS run's effective §13 per-allocation budget, in bytes.

    ``effective = min(config.runtime.memory_limit_gb * 1e9, detected
    physical RAM)`` -- a HARD clamp: the configured (or default) value can
    NEVER exceed physical RAM, however large it is set. When RAM is
    undetectable (:func:`detect_physical_ram_bytes` returns ``None``) the
    configured value is used unclamped and exactly ONE ``log.warning`` is
    emitted noting that RAM-awareness was skipped for this call.

    Returns ``(limit_bytes, source, ram_bytes)``:

    - ``limit_bytes``: the effective, per-run budget in bytes -- the value
      every guard site (via ``Tessellation.memory_limit_bytes``) should
      use.
    - ``source``: ``"config"`` when the configured value stood (RAM had
      headroom, or was undetectable), ``"ram"`` when physical RAM was the
      *smaller*, binding bound. See :func:`build_memory_guard_message`,
      which branches its advice on exactly this.
    - ``ram_bytes``: the detected physical RAM in bytes, or ``None``.

    MACHINE-DEPENDENCE WARNING: ``ram_bytes`` and ``source`` (and, when
    ``source == "ram"``, ``limit_bytes`` itself) depend on the calling
    HOST's RAM and therefore differ machine to machine for an otherwise
    identical config. Summary/log measurements may record them, while
    resolved_config.yaml retains config.runtime.memory_limit_gb unchanged.
    """
    configured_bytes = float(config.runtime.memory_limit_gb) * _GB
    ram_bytes = detect_physical_ram_bytes()
    if ram_bytes is None:
        log.warning(
            "Physical RAM could not be detected on this platform -- "
            "RAM-awareness for the §13 memory guard is skipped; using the "
            "configured runtime.memory_limit_gb=%.4g GB unclamped.",
            config.runtime.memory_limit_gb,
        )
        return configured_bytes, "config", None
    if ram_bytes < configured_bytes:
        return ram_bytes, "ram", ram_bytes
    return configured_bytes, "config", ram_bytes


def log_effective_budget(
    limit_bytes: float,
    source: str,
    ram_bytes: float | None,
    configured_gb: float,
) -> None:
    """Emit the single per-run INFO line reporting the resolved §13
    effective budget (:func:`resolve_memory_budget`'s return, unpacked).

    ``pipeline.run`` resolves the budget EXACTLY ONCE per run -- before any
    of the three tessellation-binding branches (voxel_import /
    single-crystal / flat-or-curved) constructs anything -- and calls this
    function exactly once right after, so this line appears exactly once
    in ``run.log`` regardless of which branch the run takes. Previously
    ``pipeline._stage_tessellation`` logged this inline and
    ``pipeline._apply_runtime_limits`` re-resolved (and, on the
    flat-or-curved path, therefore duplicated both this line's information
    and :func:`resolve_memory_budget`'s own undetectable-RAM warning);
    both now accept the already-resolved tuple and skip re-resolving/
    re-logging when it is given (see their docstrings), calling this
    themselves only as a fallback for a standalone caller that never went
    through ``pipeline.run``.
    """
    if source == "ram":
        # ram_bytes is guaranteed non-None here: resolve_memory_budget only
        # returns source="ram" when it detected and used a (smaller) RAM
        # figure (see its docstring).
        assert ram_bytes is not None
        log.info(
            "Memory guard (§13): %.4g GB per allocation -- clamped from "
            "configured runtime.memory_limit_gb=%.4g GB by detected "
            "physical RAM (%.4g GB).",
            limit_bytes / 1e9, configured_gb, ram_bytes / 1e9,
        )
    else:
        log.info(
            "Memory guard (§13): %.4g GB per allocation "
            "(runtime.memory_limit_gb).",
            limit_bytes / 1e9,
        )


def format_bytes_adaptive(num_bytes: float) -> str:
    """Human-readable size with ADAPTIVE units: MB (10**6 B) below 1 GB,
    GB (10**9 B, decimal -- matching how RAM/``--max-rss``/the §13 limit
    are already quoted everywhere else in this module) at or above it,
    with enough significant digits (``%.3g``) to stay actionable at any
    scale -- unlike a fixed ``.1f`` GB format, which collapses every
    sub-GB value to the useless "0.0 GB" this replaces."""
    num_bytes = max(float(num_bytes), 0.0)
    gb = num_bytes / _GB
    if gb >= 1.0:
        return f"{gb:.3g} GB"
    return f"{num_bytes / _MB:.3g} MB"


def build_memory_guard_message(
    what: str,
    est_bytes: float,
    limit_bytes: float,
    source: str,
    extra_advice: str = "",
) -> str:
    """Shared §13 guard error-message builder -- replaces the three
    copy-pasted f-strings that used to live in atoms/fill.py,
    tessellation/voxel.py and tessellation/voxel_import.py.

    Parameters
    ----------
    what : str
        Site-specific noun phrase naming WHAT allocation tripped the
        guard, e.g. ``"fill_grain: lattice grid for grain 3"`` or
        ``"Voxel grid (48, 48, 48)"`` -- this function appends the
        size/limit clause and advice.
    est_bytes, limit_bytes : float
        The estimated allocation size and the (already resolved,
        RAM-clamped -- see :func:`resolve_memory_budget`) effective limit
        it exceeds, in bytes. Formatted with adaptive units (see
        :func:`format_bytes_adaptive`) so a sub-GB limit never collapses
        to the useless "0.0 GB".
    source : str
        ``"config"`` or ``"ram"`` (:func:`resolve_memory_budget`'s
        return, normally read off ``tess.memory_limit_source``).
        ``"ram"``: physical RAM, not the configured/default value, was
        the binding, smaller bound -- the advice pivots to "this machine
        doesn't have enough RAM" (raising a config value the RAM clamp
        will just re-clamp is not actionable advice, so that sentence is
        NOT repeated in this branch). ``"config"``: the user's own
        ``runtime.memory_limit_gb`` bound the allocation with RAM
        headroom to spare -- keep the existing advice to raise it.
    extra_advice : str
        Site-specific advice appended after the source-branched sentence
        (e.g. voxel.py's "Use a smaller analysis.voxel_grid.",
        voxel_import.py's coarser-label-field advice). The "--jobs N ...
        each allocating its own grid" sentence is passed here ONLY by
        atoms/fill.py: it is the one guard site actually invoked inside
        the per-grain ``ProcessPoolExecutor`` (jobs > 1). ``build_voxel_
        grid`` (voxel.py) builds exactly one whole-box grid in the
        single-threaded driver, and voxel_import.py's margin EDT is
        likewise single-threaded -- that sentence would be simply FALSE
        at either site, which is why it is not baked into this function.
    """
    est_s = format_bytes_adaptive(est_bytes)
    limit_s = format_bytes_adaptive(limit_bytes)
    if source == "ram":
        msg = (
            f"{what} needs ~{est_s}, but this machine does not have "
            f"enough RAM (§13 memory guard): detected physical RAM is "
            f"only {limit_s}, so the configured/default "
            "runtime.memory_limit_gb was clamped down to it. Reduce the "
            "box size / grain count, or move to a machine with more RAM."
        )
    else:
        msg = (
            f"{what} needs ~{est_s} (> {limit_s} limit, §13). Raise "
            "runtime.memory_limit_gb if the machine has the RAM."
        )
    if extra_advice:
        msg = f"{msg} {extra_advice}"
    return msg


class MemoryMonitor:
    """Best-effort background sampler of the driver process's total RSS.

    Parameters
    ----------
    max_rss_gb : float | None
        Explicit WARN ceiling in GB (1 GB = 1e9 bytes). When ``None`` the
        ceiling defaults to :data:`MEMORY_SOFT_WARN_FRACTION` of physical RAM
        (auto). When physical RAM cannot be queried, no ceiling is set and the
        monitor only reports the observed peak.
    poll_interval : float
        Sampling period in seconds (default
        :data:`MEMORY_MONITOR_POLL_SECONDS`).

    Notes
    -----
    The monitor only *reads* RSS; it has no effect on the generated structure or
    on physical model data. CSV runtime observations are nondeterministic.
    Use it as a context
    manager or via :meth:`start` / :meth:`stop` (both idempotent and safe when
    psutil is absent).
    """

    def __init__(self, max_rss_gb: float | None = None,
                 poll_interval: float = MEMORY_MONITOR_POLL_SECONDS) -> None:
        self.poll_interval = poll_interval
        self.peak_bytes: int = 0
        self.tripped: bool = False
        self.auto_ceiling: bool = False
        self.max_rss_bytes: float | None = (
            float(max_rss_gb) * _GB if max_rss_gb is not None else None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._proc = psutil.Process()
            self.available = True
            if self.max_rss_bytes is None:
                total = float(psutil.virtual_memory().total)
                if total > 0:
                    self.max_rss_bytes = MEMORY_SOFT_WARN_FRACTION * total
                    self.auto_ceiling = True
        except Exception:  # psutil missing or unusable → monitor is inert
            self._proc = None
            self.available = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> MemoryMonitor:
        """Begin background sampling (no-op if psutil is absent)."""
        if self.available and self._thread is None:
            self._sample()  # baseline
            self._thread = threading.Thread(
                target=self._loop, name="grainsmith-rss", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling and take a final reading (idempotent)."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        self._sample()

    def __enter__(self) -> MemoryMonitor:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()  # returns None → never suppresses an exception

    # -- sampling ------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            self._sample()

    def _sample(self) -> None:
        if self._proc is None:
            return
        try:
            rss = int(self._proc.memory_info().rss)
        except Exception:  # process gone / platform quirk — stay silent
            return
        if rss > self.peak_bytes:
            self.peak_bytes = rss
        if (self.max_rss_bytes is not None and rss > self.max_rss_bytes
                and not self.tripped):
            self.tripped = True
            # Real-time forewarning from the sampler thread: land the WARN in
            # run.log the moment the soft ceiling is crossed, so an impending
            # OOM-kill is announced even if the run dies before log_summary().
            which = "system-RAM auto" if self.auto_ceiling else "--max-rss"
            log.warning(
                "peak driver RSS crossed the %s ceiling (%.2f GB) at %.2f GB — "
                "reduce the problem size or raise the ceiling; the "
                "per-allocation guard does not bound total RSS.",
                which, self.max_rss_bytes / _GB, rss / _GB)

    # -- reporting -----------------------------------------------------------

    @property
    def peak_rss_bytes(self) -> float | None:
        """Observed peak RSS in bytes, or ``None`` when monitoring is inert."""
        if not self.available or self.peak_bytes <= 0:
            return None
        return float(self.peak_bytes)

    def summary(self) -> str | None:
        """One-line human summary, or ``None`` when monitoring is inert."""
        if not self.available:
            return None
        peak_gb = self.peak_bytes / _GB
        if self.max_rss_bytes is None:
            return f"peak driver RSS {peak_gb:.2f} GB"
        limit_gb = self.max_rss_bytes / _GB
        which = "system-RAM auto-ceiling" if self.auto_ceiling else "--max-rss"
        level = "WARN" if self.tripped else "ok"
        return (f"{level}: peak driver RSS {peak_gb:.2f} GB "
                f"vs {which} {limit_gb:.2f} GB")

    def report(self) -> dict[str, object]:
        """Current measured RSS and warning settings for a run's CSV summary.

        These are observations, not deterministic model data or a hard RSS cap.
        The driver excludes child-process RSS, matching this monitor's scope.
        """
        self._sample()
        peak = self.peak_rss_bytes
        return {
            "monitor_available": self.available,
            "peak_driver_rss_gb": None if peak is None else peak / _GB,
            "rss_warn_limit_gb": (None if self.max_rss_bytes is None else
                                  self.max_rss_bytes / _GB),
            "rss_warn_limit_source": ("system_ram_auto" if self.auto_ceiling else
                                      "--max-rss" if self.max_rss_bytes is not None else
                                      "unavailable"),
            "rss_limit_kind": "warning_only",
            "rss_warn_tripped": self.tripped,
            "rss_scope": "driver_only; sampled through primary outputs before summary",
        }

    def log_summary(self) -> None:
        """Emit the summary at WARNING level if tripped, else INFO."""
        s = self.summary()
        if s is None:
            return
        if self.tripped:
            log.warning("%s — reduce the problem size or raise the ceiling; "
                        "the per-allocation guard does not bound total RSS.", s)
        else:
            log.info("%s", s)
