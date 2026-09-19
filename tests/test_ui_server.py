"""Endpoint contract tests for the grainsmith studio FastAPI backend.

Skipped entirely when the optional ``[ui]`` extra (fastapi / httpx) is absent,
so the core test run stays dependency-free.  Each test pins that an endpoint
delegates to the pure module it wraps — the server adds no second code path.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from grainsmith.ui.formspec import build_form, iter_field_names  # noqa: E402
from grainsmith.ui.schema_json import schema_to_json  # noqa: E402
from grainsmith.ui.server import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _field_names(doc: dict) -> set[str]:
    """Every field/section name reachable in a /api/schema payload."""
    names: set[str] = set()

    def walk_model(model: dict) -> None:
        for f in model.get("fields", []):
            names.add(f["name"])
            if f.get("children"):
                walk_model(f["children"])
            if f.get("element_model"):
                walk_model(f["element_model"])

    for section in doc["sections"]:
        names.add(section["name"])
        walk_model(section)
    return names


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_schema_endpoint_is_serialized_form(client: TestClient) -> None:
    resp = client.get("/api/schema")
    assert resp.status_code == 200
    # The endpoint serves exactly schema_to_json() (JSON round-trip identical).
    assert resp.json() == json.loads(json.dumps(schema_to_json()))


def test_schema_endpoint_covers_every_field(client: TestClient) -> None:
    """The no-silent-field-drop guarantee, re-pinned at the HTTP boundary."""
    doc = client.get("/api/schema").json()
    assert iter_field_names(build_form()) <= _field_names(doc)


def test_presets_lists_shipped_examples(client: TestClient) -> None:
    presets = client.get("/api/presets").json()["presets"]
    assert isinstance(presets, list)
    # The repo ships examples (editable install); the canonical first run is one.
    assert "b2_nial_flat" in presets


def test_preset_load_round_trips_through_validate(client: TestClient) -> None:
    preset = client.get("/api/presets/b2_nial_flat")
    assert preset.status_code == 200
    body = preset.json()
    assert "yaml" in body and isinstance(body["values"], dict)
    # The preset's own values must validate (it came through the engine).
    ok = client.post("/api/validate", json={"values": body["values"]})
    assert ok.status_code == 200 and ok.json()["ok"] is True


def test_unknown_preset_is_404(client: TestClient) -> None:
    assert client.get("/api/presets/does_not_exist_xyz").status_code == 404


def test_validate_rejects_unknown_key_with_config_error(client: TestClient) -> None:
    resp = client.post("/api/validate", json={"values": {"not_a_field": 1}})
    assert resp.status_code == 422
    assert "message" in resp.json()


def test_yaml_to_values_round_trip(client: TestClient) -> None:
    yaml_text = client.get("/api/presets/b2_nial_flat").json()["yaml"]
    resp = client.post("/api/yaml/to-values", json={"yaml": yaml_text})
    assert resp.status_code == 200
    values = resp.json()["values"]
    assert isinstance(values, dict)
    # Those values validate again — a clean round-trip.
    assert client.post("/api/validate", json={"values": values}).status_code == 200


def test_outputs_404_on_missing_dir(client: TestClient, tmp_path) -> None:
    missing = tmp_path / "nope"
    assert client.get("/api/outputs", params={"dir": str(missing)}).status_code == 404


def test_outputs_reads_recorded_microstructure(client: TestClient, tmp_path) -> None:
    # A minimal recorded artifact is enough to exercise the read path.
    (tmp_path / "microstructure.json").write_text(
        json.dumps(
            {
                "provenance": {"title": "demo", "grainsmith_version": "1.1.0"},
                "gates": [{"gate": "G1", "passed": True, "measured": "ok", "message": "ok"}],
                "statistics": {},
                "grains": [{"volume_A3": 1234.5}],
            }
        ),
        encoding="utf-8",
    )
    resp = client.get("/api/outputs", params={"dir": str(tmp_path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "demo"
    assert body["present"]["microstructure_json"] is True
    assert body["grain_volumes_A3"] == [1234.5]
    assert body["gates"][0]["gate"] == "G1"


# ---------------------------------------------------------------------------
# /api/run — streaming in-app run
# ---------------------------------------------------------------------------


def _minimal_run_values(outdir) -> dict:
    """Smallest fast polycrystal (4-grain FCC Cu, 40^3 A)."""
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


def _parse_sse(line_iter) -> list[tuple[str, dict]]:
    """Pair each ``event:`` line with its following ``data:`` JSON."""
    events: list[tuple[str, dict]] = []
    current: str | None = None
    for raw in line_iter:
        line = raw.rstrip("\r")
        if line.startswith("event:"):
            current = line[len("event:") :].strip()
        elif line.startswith("data:"):
            events.append((current or "", json.loads(line[len("data:") :].strip())))
            current = None
    return events


def test_run_notice(client: TestClient) -> None:
    resp = client.get("/api/run/notice")
    assert resp.status_code == 200
    assert "memory" in resp.json()["notice"].lower()


def test_run_invalid_config_is_422_before_stream(client: TestClient) -> None:
    resp = client.post("/api/run", json={"values": {"not_a_field": 1}, "jobs": 1})
    assert resp.status_code == 422


def test_run_conflict_when_locked(client: TestClient, tmp_path) -> None:
    from grainsmith.ui.run_stream import RUN_LOCK

    assert RUN_LOCK.acquire(blocking=False)
    try:
        resp = client.post(
            "/api/run",
            json={"values": _minimal_run_values(tmp_path / "locked"), "jobs": 1},
        )
        assert resp.status_code == 409
    finally:
        RUN_LOCK.release()


def test_run_streams_to_completion(client: TestClient, tmp_path) -> None:
    values = _minimal_run_values(tmp_path / "out")
    with client.stream(
        "POST", "/api/run", json={"values": values, "jobs": 1}
    ) as resp:
        assert resp.status_code == 200
        events = _parse_sse(resp.iter_lines())

    kinds = [e for e, _ in events]
    assert "run-start" in kinds
    finals = [d for e, d in events if e == "final-result"]
    assert finals, f"no final-result; saw {kinds}"
    final = finals[0]
    assert final["n_grains"] == 4
    assert final["n_atoms"] > 0
    assert final["gates_passed"] == final["gates_total"]
    assert final["failed"] == []


# ---------------------------------------------------------------------------
# Lock lifetime across client disconnect — Fix #16
# ---------------------------------------------------------------------------


def test_lock_held_after_disconnect_released_when_worker_finishes(
    monkeypatch, tmp_path
) -> None:
    """Lock must NOT be released when the SSE client disconnects mid-stream.

    Scenario: the SSE generator has been started (run-start yielded, worker
    thread running), then the client disconnects (generator.close()).  The lock
    must remain held until the worker thread's pipeline_run() returns; only then
    is it released.

    The test monkeypatches pipeline_run with a stub that blocks until explicitly
    unblocked, letting us control the race precisely.

    Because the generator's polling loop blocks in time.sleep(), we drive the
    generator in a background thread so the main test thread can call .close()
    without deadlocking.
    """
    import threading
    import time as _time
    import grainsmith.ui.run_stream as rs

    outdir = tmp_path / "disconnect_test"
    outdir.mkdir()

    # --- controllable blocking stub for pipeline_run ---
    worker_may_finish = threading.Event()
    worker_started = threading.Event()

    def _fake_pipeline_run(config, *, jobs):  # noqa: ARG001
        worker_started.set()
        worker_may_finish.wait(timeout=10)
        return None

    monkeypatch.setattr(rs, "pipeline_run", _fake_pipeline_run)

    from grainsmith.ui.presets import load_preset
    from grainsmith.config.schema import RunConfig

    config, _ = load_preset("b2_nial_flat")
    config = RunConfig(
        **{
            **config.model_dump(),
            "output": config.output.model_copy(update={"directory": str(outdir)}),
        }
    )

    lock = threading.Lock()
    assert lock.acquire(blocking=False), "test setup: lock should be free"

    gen = rs.run_events(config, jobs=1, lock=lock)

    # The first next() returns the run-start event; at that point the generator
    # is paused just before thread.start().
    first_event = next(gen)
    assert "run-start" in first_event

    # Drive the generator in a background thread.  The driver calls next() to
    # start the polling loop (which triggers thread.start()), then closes the
    # generator once the main thread signals it — simulating a client disconnect.
    # We must close from the *same* thread that is iterating (Python forbids
    # gen.close() while another thread is inside next(gen)).
    close_signal = threading.Event()
    gen_advanced = threading.Event()  # set when driver is past thread.start()
    gen_exception: list[BaseException] = []

    def _drive_gen() -> None:
        try:
            # One next() call advances past thread.start() into the polling loop;
            # it will block for up to 0.15 s (time.sleep in the loop), which is
            # fine — we don't need to wait for it before signalling close.
            next(gen)
            gen_advanced.set()
            # Wait for the main thread to signal that the worker is running
            close_signal.wait(timeout=10)
        except StopIteration:
            gen_advanced.set()
        except Exception as exc:  # pragma: no cover
            gen_exception.append(exc)
            gen_advanced.set()
        finally:
            # Close from the driver (the thread that was iterating the generator)
            try:
                gen.close()
            except Exception:  # pragma: no cover
                pass

    driver = threading.Thread(target=_drive_gen, daemon=True)
    driver.start()

    # Wait until the _fake_pipeline_run has been entered by the worker thread
    assert worker_started.wait(timeout=5), "worker thread did not start in time"

    # Signal the driver to close the generator (simulating client disconnect)
    close_signal.set()

    # Wait for the driver to complete the close
    driver.join(timeout=5.0)
    assert not driver.is_alive(), "driver thread did not finish in time"

    # --- KEY assertion: lock must still be held (worker still blocking) ---
    assert lock.locked(), (
        "Bug: lock was released on client disconnect while worker still running"
    )
    assert not gen_exception, f"generator raised unexpectedly: {gen_exception}"

    # Unblock the worker; it should release the lock in its own finally block
    worker_may_finish.set()

    # Poll up to 2 s for lock release
    for _ in range(40):
        if not lock.locked():
            break
        _time.sleep(0.05)

    assert not lock.locked(), (
        "Bug: lock was not released after worker thread finished"
    )


def test_lock_released_when_worker_start_fails(monkeypatch, tmp_path) -> None:
    """If thread.start() itself raises, the generator must still release the lock.

    Control flow: first next() yields run-start (pauses there), second next()
    resumes into thread.start() which raises RuntimeError, propagating out of
    next(). The generator's finally block must catch that and release the lock
    (worker never ran, so the edge-case guard fires).
    """
    import threading
    import grainsmith.ui.run_stream as rs

    outdir = tmp_path / "start_fail_test"
    outdir.mkdir()

    # Make pipeline_run a no-op (unreachable — thread.start() will raise first)
    monkeypatch.setattr(rs, "pipeline_run", lambda *a, **kw: None)

    # Make thread.start() raise only for the run worker thread
    original_start = threading.Thread.start

    def _bad_start(self):
        if self.name == "grainsmith-run":
            raise RuntimeError("simulated thread start failure")
        return original_start(self)  # pragma: no cover

    monkeypatch.setattr(threading.Thread, "start", _bad_start)

    from grainsmith.ui.presets import load_preset
    from grainsmith.config.schema import RunConfig

    config, _ = load_preset("b2_nial_flat")
    config = RunConfig(
        **{
            **config.model_dump(),
            "output": config.output.model_copy(update={"directory": str(outdir)}),
        }
    )

    lock = threading.Lock()
    assert lock.acquire(blocking=False)

    gen = rs.run_events(config, jobs=1, lock=lock)

    # First next() → runs up to the yield and returns run-start
    first = next(gen)
    assert "run-start" in first

    # Second next() → resumes after the yield, hits thread.start() → RuntimeError
    # propagates out; the generator's finally fires (worker_started=False → release)
    with pytest.raises(RuntimeError, match="simulated thread start failure"):
        next(gen)

    # Lock must have been released by the generator's finally (worker never ran)
    assert not lock.locked(), (
        "Bug: lock not released when thread.start() failed before worker ran"
    )


# ---------------------------------------------------------------------------
# /api/pick-dir — native folder dialog (mocked; no real GUI in tests)
# ---------------------------------------------------------------------------


def test_pick_dir_returns_native_choice(client: TestClient, monkeypatch) -> None:
    import grainsmith.ui.server as srv

    monkeypatch.setattr(srv, "_native_pick_directory", lambda: "/picked/run")
    resp = client.get("/api/pick-dir")
    assert resp.status_code == 200
    assert resp.json() == {"dir": "/picked/run"}


def test_pick_dir_empty_when_cancelled(client: TestClient, monkeypatch) -> None:
    import grainsmith.ui.server as srv

    monkeypatch.setattr(srv, "_native_pick_directory", lambda: "")
    resp = client.get("/api/pick-dir")
    assert resp.status_code == 200
    assert resp.json() == {"dir": ""}
