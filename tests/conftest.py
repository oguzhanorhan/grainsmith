"""pytest configuration and shared fixtures for the grainsmith test suite.

``tmp_path`` is provided by pytest natively (no fixture needed here).
Add project-level fixtures as phases are implemented.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks (avoids PytestUnknownMarkWarning).

    ``fig2_validation``: the expensive, opt-in real-system numba-vs-numpy
    owns() mask-equality replay against the shipped paper Fig. 2 curved
    system (tests/test_owns_kernel.py) — skipped by default, enabled via
    GRAINSMITH_RUN_FIG2_VALIDATION=1.
    """
    config.addinivalue_line(
        "markers",
        "fig2_validation: expensive real-system validation, opt-in via "
        "GRAINSMITH_RUN_FIG2_VALIDATION=1 (deselected by default)",
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def examples_dir(project_root: Path) -> Path:
    """Path to the examples/ directory."""
    return project_root / "examples"


@pytest.fixture(scope="session")
def b2_nial_yaml(examples_dir: Path) -> Path:
    """Path to the B2 NiAl flat-boundary example config."""
    return examples_dir / "basics" / "b2_nial_flat.yaml"


@pytest.fixture(scope="session")
def fcc_cu_curved_yaml(examples_dir: Path) -> Path:
    """Path to the FCC Cu curved-warp example config."""
    return examples_dir / "grain_geometry" / "fcc_cu_curved_warp.yaml"


@pytest.fixture(scope="session")
def hcp_ti_yaml(examples_dir: Path) -> Path:
    """Path to the HCP Ti thin-film example config."""
    return examples_dir / "basics" / "hcp_ti_thin_film.yaml"


@pytest.fixture(scope="session")
def bicrystal_yaml(examples_dir: Path) -> Path:
    """Path to the Cu bicrystal Σ5 example config."""
    return examples_dir / "basics" / "bicrystal_fixed_orient.yaml"
