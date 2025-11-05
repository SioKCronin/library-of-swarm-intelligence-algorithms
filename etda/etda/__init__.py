"""ETDA: Evolutionary Topological Data Analysis

Integration of swarm optimization with giotto-tda for high-dimensional
optimization problems, especially health data applications.
"""

from .core import ETDAOptimizer, TDAProcessor, ManifoldReducer
from .roi import create_high_dim_roi, HighDimROI

__all__ = [
    "ETDAOptimizer",
    "TDAProcessor",
    "ManifoldReducer",
    "create_high_dim_roi",
    "HighDimROI",
]

__version__ = "0.1.0"

