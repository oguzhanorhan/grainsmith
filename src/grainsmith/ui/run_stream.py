"""Server-sent-events (SSE) streaming for in-app pipeline runs.

The engine's :func:`grainsmith.pipeline.run` is a single blocking call with no
progress callback, so to stream progress we run it in a daemon worker thread and
tail the ``run.log`` file the pipeline writes into the run's output directory.
Stage events are **best-effort**, derived from recognisable log-line prefixes;
the authoritative results (atom / grain / boundary counts and the QA-gate
verdicts) come from the returned :class:`~grainsmith.pipeline.RunResult` once the
worker thread joins.

A module-level lock enforces ONE in-app run at a time: the pipeline reconfigures
the root logger with ``logging.basicConfig(force=True)``, so two concurrent runs
would clobber each other's logging.  The HTTP layer returns 409 when the lock is
already held.

Pure of FastAPI: this module only yields SSE-formatted strings, so it stays
unit-testable; the server wraps the generator in a ``StreamingResponse``.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from grainsmith.config.schema import RunConfig
from grainsmith.pipeline import run as pipeline_run

# Single in-app run at a time (the pipeline's basicConfig(force=True) makes the
# root-logger setup process-global).  The server acquires this non-blockingly.
RUN_LOCK = threading.Lock()

# Pre-run caution surfaced to the UI before an in-app run starts.
RUN_MEMORY_NOTICE = (
    "An in-app run blocks until it finishes and holds the whole structure in "
    "memory. For large / flagship configs use the CLI instead "
    "(`grainsmith generate config.yaml`); the engine's per-allocation memory "
    "guard may abort very large runs."
)

# Best-effort stage markers recognised in run.log message text.
_STAGE_MARKERS: dict[str, str] = {
    "crystal:": "crystal",
    "sdot:": "size_distribution",
    "voxel_import:": "voxel_import",
    "fill:": "fill",
    "overlap:": "overlap",
    "phases:": "phases",
    "done:": "done",
}


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one SSE message frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _json_scalar(value: Any) -> Any:
    """JSON-safe scalar for a gate's ``measured`` field (NaN/Inf -> None)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, bool | int | str):
        return value
    return str(value)


def _read_new_lines(path: Path, pos: int) -> tuple[int, list[str]]:
    """Read complete new lines from *path* starting at byte offset *pos*.

    Returns the advanced offset and the new, non-blank lines.  Reads in binary
    and only consumes up to the last newline, so a partially-written trailing
    line is never emitted (and then re-read in full on the next poll).
    """
    if not path.is_file():
        return pos, []
    try:
        with path.open("rb") as fh:
            fh.seek(pos)
            data = fh.read()
    except OSError:
        return pos, []
    nl = data.rfind(b"\n")
    if nl == -1:
        return pos, []
    consumed = data[: nl + 1]
    text = consumed.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return pos + len(consumed), lines


def _stage_for(line: str) -> str | None:
    """Map a log line to a coarse stage name, or ``None``."""
    for marker, stage in _STAGE_MARKERS.items():
        if marker in line:
            return stage
    return None


def run_events(config: RunConfig, jobs: int, lock: threading.Lock) -> Iterator[str]:
    """Yield SSE frames for a pipeline run; release *lock* when finished.

    Emits ``run-start``, a stream of ``log-line`` / ``stage`` events while the
    run proceeds, then per-gate ``gate`` events and a terminal ``final-result``
    (or ``error``).  *lock* must already be held by the caller.

    Lock lifetime
    -------------
    The worker thread is the **sole** owner of the lock once it has started:
    it releases the lock in its own ``finally`` block after ``pipeline_run``
    returns or raises.  This means a client disconnect (``GeneratorExit``) does
    NOT prematurely release the lock — the 409 guard in the server correctly
    blocks a second run until the real work ends.

    Edge case: if ``thread.start()`` raises before the worker ever runs, the
    worker's ``finally`` will never execute, so the generator releases the lock
    itself in that narrow failure path.
    """
    outdir = Path(config.output.directory).resolve()
    log_path = outdir / "run.log"
    result_box: dict[str, Any] = {}
    _worker_started = False

    def _worker() -> None:
        try:
            result_box["result"] = pipeline_run(config, jobs=jobs)
        except Exception as exc:  # reported to the client as an SSE 'error' event
            result_box["error"] = exc
        finally:
            lock.release()  # sole release point once the worker is running

    try:
        yield _sse("run-start", {"outdir": str(outdir), "jobs": jobs})
        thread = threading.Thread(target=_worker, name="grainsmith-run", daemon=True)
        thread.start()
        _worker_started = True

        seen: set[str] = set()
        pos = 0
        while True:
            alive = thread.is_alive()
            pos, lines = _read_new_lines(log_path, pos)
            for line in lines:
                yield _sse("log-line", {"text": line})
                stage = _stage_for(line)
                if stage and stage not in seen:
                    seen.add(stage)
                    yield _sse("stage", {"stage": stage})
            if not alive and not lines:
                break
            if not lines:
                yield ": keep-alive\n\n"
            time.sleep(0.15)

        thread.join(timeout=2.0)

        if "error" in result_box:
            exc = result_box["error"]
            yield _sse("error", {"error": type(exc).__name__, "message": str(exc)})
            return
        result = result_box.get("result")
        if result is None:
            yield _sse(
                "error", {"error": "RuntimeError", "message": "run produced no result"}
            )
            return

        gate_results = result.gates.results()
        for gr in gate_results:
            yield _sse(
                "gate",
                {
                    "gate": gr.gate,
                    "passed": bool(gr.passed),
                    "measured": _json_scalar(gr.measured),
                    "message": gr.message,
                },
            )
        failed = [gr.gate for gr in gate_results if not gr.passed]
        yield _sse(
            "final-result",
            {
                "n_atoms": len(result.atoms),
                "n_grains": result.tess.n_grains,
                "n_boundaries": len(result.boundary_reports),
                "gates_passed": sum(1 for gr in gate_results if gr.passed),
                "gates_total": len(gate_results),
                "failed": failed,
                "total_s": float(result.timings.get("total", 0.0)),
                "outdir": str(result.outdir),
            },
        )
    finally:
        # Only release here when the worker never started (thread.start() raised
        # before the worker's own finally could take ownership).  In all normal
        # paths — including client disconnect — the worker releases the lock.
        if not _worker_started:
            lock.release()
