"""Shared writer infrastructure: provenance header + deterministic formatting.

Every output file carries a provenance header where the format allows
comments (§8): ``grainsmith <version> | <ISO timestamp> | seed=<int> |
config sha256=<12 hex> | provenance sha256=<12 hex>``.  CSV files are the
exception — RFC 4180 has no comment syntax and §8.3 pins the exact header
row, so CSVs carry no provenance line (the same information lives in
summary.csv rows).

All writers open files with ``newline="\\n"`` so output is byte-identical
across platforms (Windows text mode would otherwise translate line ends),
which the end-to-end reproducibility test (§10) relies on.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass(frozen=True)
class Provenance:
    """Run identification embedded in every output header (§8).

    Two digests, both stored full-length (64 hex) and displayed truncated:

    ``config_sha256``
        SHA-256 of the canonical resolved config ONLY (config.resolve.
        config_sha256). Identifies the *input*. Two different grainsmith
        versions run on the same config produce the same value.
    ``provenance_sha256``
        SHA-256 of a payload that CONTAINS the producing grainsmith version
        (grainsmith.provenance.provenance_payload). Identifies *input +
        producer*, so an output's version claim is cryptographically bound to
        its digest and cannot be edited after the fact without detection
        (``grainsmith verify``).
    """
    version: str
    timestamp_iso: str      # e.g. "2026-06-11T12:00:00Z" (UTC)
    seed: int
    config_sha256: str      # full 64 hex — resolved-config digest
    provenance_sha256: str  # full 64 hex — version-bound digest
    title: str = ""

    @property
    def config_sha12(self) -> str:
        """Display form of ``config_sha256`` (first 12 hex)."""
        return self.config_sha256[:12]

    @property
    def provenance_sha12(self) -> str:
        """Display form of ``provenance_sha256`` (first 12 hex)."""
        return self.provenance_sha256[:12]

    def line(self) -> str:
        """The §8 provenance string (no comment marker — caller prefixes)."""
        return (f"grainsmith {self.version} | {self.timestamp_iso} | "
                f"seed={self.seed} | config sha256={self.config_sha12} | "
                f"provenance sha256={self.provenance_sha12}")


@contextmanager
def atomic_writer(path: Path, encoding: str = "utf-8",
                  newline: str = "\n") -> Iterator[IO[str]]:
    """Open a sibling ``.tmp`` file, yield its handle, then atomically
    ``os.replace`` it onto *path* on clean exit.

    On any exception the partial temp file is removed and *path* is left
    untouched — so a mid-write failure (a worker-pool crash, a full disk, an
    OOM-kill or Ctrl-C) can never leave a truncated file at a user-facing output
    path, nor silently overwrite a previously good file from an earlier run.
    ``os.replace`` is atomic within a filesystem, which every writer here
    satisfies (the temp file is a sibling of the target).  Gate G10 validates
    LAMMPS output only after a successful write, so it never inspects this
    failure path; this guard closes it.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    fh = tmp.open("w", encoding=encoding, newline=newline)
    try:
        yield fh
        # fsync before the rename: os.replace is atomic with respect to other
        # PROCESSES, but on a crash or power loss the rename can reach disk
        # while the data behind it has not, leaving a correctly-named empty or
        # truncated file. Flushing the file's own pages first closes that
        # window. Costs one syscall per output file.
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()
        os.replace(tmp, path)
    except BaseException:
        fh.close()
        tmp.unlink(missing_ok=True)
        raise


def fmt(x: float) -> str:
    """Shortest decimal representation that round-trips to the same float64.

    Python's float repr is the shortest string guaranteed to parse back to
    the identical double — this makes text outputs both exact (gate G10
    bounds round-trip) and deterministic (byte-identical re-runs).
    """
    return repr(float(x))


# Streaming write: atom rows are formatted in contiguous
# row-index chunks so the writer never materialises a full wrapped-position
# copy or the whole formatted file in memory, and the per-row repr()
# formatting can fan out across worker processes.  Chunking the *same* serial
# loop at row boundaries makes the output bytes identical for any chunk size
# and any number of workers (the byte-identical determinism invariant, §1).
WRITE_CHUNK = 250_000


def drain_in_order(fh, n: int, chunk: int, window: int, submit) -> None:
    """Write per-chunk text to *fh* in ascending chunk order.

    ``submit(a, b)`` returns a :class:`concurrent.futures.Future` that yields
    the formatted text for rows ``[a, b)``.  At most *window* futures are kept
    outstanding, so peak memory stays ~O(window) chunks regardless of file
    size while results are still written strictly in order — the join is
    deterministic and independent of which worker finishes first.
    """
    from collections import deque

    if window < 1:
        raise ValueError(f"drain_in_order window must be >= 1, got {window}")
    starts = iter(range(0, n, chunk))
    futures: deque = deque()
    for _ in range(window):
        a = next(starts, None)
        if a is None:
            break
        futures.append(submit(a, min(a + chunk, n)))
    while futures:
        fh.write(futures.popleft().result())
        a = next(starts, None)
        if a is not None:
            futures.append(submit(a, min(a + chunk, n)))
