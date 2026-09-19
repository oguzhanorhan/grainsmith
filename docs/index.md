# grainsmith documentation

**grainsmith** generates space-group-aware atomistic polycrystals for molecular
dynamics — with statistically controlled microstructure (grain size, texture,
grain-boundary network, boundary roughness), construction-time QA gates, and
reproducible, publication-grade outputs. Current output target: LAMMPS.

This is the full documentation set. It serves both first-time users (install →
first model → reading the outputs) and advanced users and developers (the
physics, the module architecture, the programmatic API, and how to extend the
code).

## Start here

- [**User manual**](manual.md) — install, quickstart, the CLI, and how a run is
  built. Read this first.
- [**Configuration reference**](config_reference.md) — every config field, its
  type, default, and constraints (auto-generated from the schema).

## Understand the system

- [**Architecture**](architecture.md) — the pipeline stages, the data objects
  that flow between them, the package structure, and the module dependency
  graph.
- [**Grain geometry**](geometry.md) — the flat / power / curved /
  self-affine (perturbed-distance) / weighted / voxel-import backends, the
  exact parameters that control each, and the construction conditions they
  must satisfy.
- [**Physics & conventions**](physics.md) — the math and the coordinate,
  orientation, and unit conventions the code commits to.

## Reference

- [**QA gates (G1–G26)**](gates.md) — every construction-time correctness gate,
  what it checks, and what a failure means.
- [**Outputs**](outputs.md) — every file a run writes and every column in it.

## For developers

- [**Developer guide**](developer.md) — the repository layout, the `run()`
  entry point, the `RunResult` contract, the public API per subpackage, the
  determinism contract and the parallel pattern that implements it, how to
  add a tessellation backend, orientation sampler, or output writer, and the
  test suite's own conventions.

## Help

- [**FAQ & troubleshooting**](faq.md) — common errors and what to do about them.

*grainsmith is reproducible by design: every stochastic step records an explicit
RNG seed, and the same seed produces a bit-identical model (see the manual's
"Determinism and provenance" for what that does and does not cover across
machines, and for `grainsmith verify`). Every physical constraint is enforced by
a QA gate at construction time — a model that would be physically invalid is
never written.*
