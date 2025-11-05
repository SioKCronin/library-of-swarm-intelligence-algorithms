"""
Region of Interest (ROI) generation for high-dimensional spaces.

Provides utilities for creating large-scale multidimensional regions
of interest for health data and other applications.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class HighDimROI:
    """High-dimensional region of interest."""
    data: np.ndarray
    bounds: list[tuple[float, float]]
    n_samples: int
    n_features: int
    seed: Optional[int] = None


def create_high_dim_roi(
    n_samples: int = 1000,
    n_features: int = 50,
    bounds: Optional[list[tuple[float, float]]] = None,
    distribution: str = "uniform",
    correlation: Optional[float] = None,
    seed: Optional[int] = None,
) -> HighDimROI:
    """Create a high-dimensional region of interest.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features/dimensions
        bounds: Bounds for each dimension (default: (-1, 1) for all)
        distribution: Distribution type ("uniform", "normal", "multimodal")
        correlation: Correlation coefficient for correlated features (0-1)
        seed: Random seed
        
    Returns:
        HighDimROI object with data and metadata
    """
    rng = np.random.RandomState(seed)
    
    if bounds is None:
        bounds = [(-1.0, 1.0)] * n_features
    
    # Generate data based on distribution
    if distribution == "uniform":
        data = np.array([
            rng.uniform(lo, hi, n_samples)
            for lo, hi in bounds
        ]).T
    
    elif distribution == "normal":
        # Normal distribution centered in bounds
        centers = [(lo + hi) / 2 for lo, hi in bounds]
        scales = [(hi - lo) / 6 for lo, hi in bounds]  # 3-sigma rule
        
        data = np.array([
            rng.normal(center, scale, n_samples)
            for center, scale in zip(centers, scales)
        ]).T
        
        # Clamp to bounds
        for i, (lo, hi) in enumerate(bounds):
            data[:, i] = np.clip(data[:, i], lo, hi)
    
    elif distribution == "multimodal":
        # Multiple modes for each dimension
        n_modes = 3
        data = np.zeros((n_samples, n_features))
        
        for j, (lo, hi) in enumerate(bounds):
            mode_positions = np.linspace(lo, hi, n_modes)
            mode_weights = rng.dirichlet([1] * n_modes)
            mode_scales = [(hi - lo) / 10] * n_modes
            
            # Sample from mixture
            for i in range(n_samples):
                mode = rng.choice(n_modes, p=mode_weights)
                data[i, j] = rng.normal(
                    mode_positions[mode],
                    mode_scales[mode]
                )
                data[i, j] = np.clip(data[i, j], lo, hi)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Add correlation if specified
    if correlation is not None and 0 < correlation < 1:
        # Create correlation matrix
        corr_matrix = np.eye(n_features) * (1 - correlation) + correlation
        # Cholesky decomposition for correlated sampling
        L = np.linalg.cholesky(corr_matrix)
        # Apply transformation (normalize first)
        data_normalized = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)
        data = (L @ data_normalized.T).T
        # Rescale back to original bounds
        for i, (lo, hi) in enumerate(bounds):
            data[:, i] = np.interp(
                data[:, i],
                (data[:, i].min(), data[:, i].max()),
                (lo, hi)
            )
    
    return HighDimROI(
        data=data,
        bounds=bounds,
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
    )

