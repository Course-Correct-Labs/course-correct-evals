"""
Course Correct Labs Reasoning Stability Observatory

A unified analysis system for synthesizing CCL empirical study data.
"""

__version__ = "0.1.0"

from .importers import (
    MirrorLoopImporter,
    ConfabulationImporter,
    ViolationStateImporter,
    EchoChamberImporter,  # noncanonical/opt-in — not part of the Observatory's canonical study set
)

from .analysis import CrossStudyAnalysis

__all__ = [
    "MirrorLoopImporter",
    "ConfabulationImporter",
    "ViolationStateImporter",
    "EchoChamberImporter",
    "CrossStudyAnalysis",
]
