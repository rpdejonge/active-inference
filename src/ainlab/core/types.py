"""
Shared type definitions used across the project.
"""

from typing import NewType
import numpy as np

StateIndex = NewType("StateIndex", int)
ActionIndex = NewType("ActionIndex", int)
ObservationIndex = NewType("ObservationIndex", int)

ProbabilityArray = np.ndarray
PolicyArray = np.ndarray