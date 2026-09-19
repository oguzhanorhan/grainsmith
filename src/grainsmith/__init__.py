"""grainsmith — A Generator of Polycrystalline Models for Atomistic
Simulations with Statistical and Grain-Boundary Morphology Control."""

__version__ = "1.1.0"


def __getattr__(name: str):  # noqa: ANN001, ANN202 — lazy public-API re-export
    """Lazy-load top-level public symbols to avoid import-chain issues."""
    if name == "run":
        from grainsmith.pipeline import run  # noqa: F401
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
