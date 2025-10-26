"""
Environment bootstrap utilities mirroring `setup.R`.

This module ensures required output directories exist and offers helpers
to validate that the Python dependency stack is installed.  It mirrors
the side effects of the original R script without installing packages
automatically.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable

DEFAULT_DIRECTORIES = ("figures", "tables", "results")

# The Python equivalents of the R packages installed in `setup.R`.
REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "pytest",
)


def ensure_directories(base_path: Path | str = ".",
                       directories: Iterable[str] = DEFAULT_DIRECTORIES) -> None:
    """
    Create the directories used for generated artifacts if they do not yet exist.
    """
    base_path = Path(base_path)
    for name in directories:
        target = base_path / name
        target.mkdir(parents=True, exist_ok=True)


def missing_python_packages(packages: Iterable[str] = REQUIRED_PACKAGES) -> list[str]:
    """
    Return the subset of package names that cannot be imported.
    """
    missing = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ModuleNotFoundError:
            missing.append(package)
    return missing


def assert_dependencies(packages: Iterable[str] = REQUIRED_PACKAGES) -> None:
    """
    Raise a RuntimeError if any required dependency is missing.
    """
    missing = missing_python_packages(packages)
    if missing:
        raise RuntimeError(
            "Missing required Python packages: {}".format(", ".join(sorted(missing)))
        )
