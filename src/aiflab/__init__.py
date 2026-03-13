"""
Active Inference Lab

Core research framework for building Active Inference agents
and running reproducible experiments.
"""

from importlib.metadata import version

__all__ = ["__version__"]

try:
    __version__ = version("active-inference-lab")
except Exception:
    __version__ = "0.0.0"