"""
Project path helpers.
"""

from pathlib import Path


def project_root() -> Path:
    """
    Returns project root directory.

    Assumes src layout:
    repo/
        src/
            ailab/
    """
    return Path(__file__).resolve().parents[3]


def outputs_dir() -> Path:
    p = project_root() / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def experiments_dir() -> Path:
    p = outputs_dir() / "experiments"
    p.mkdir(parents=True, exist_ok=True)
    return p