# ETDA: Evolutionary Topological Data Analysis

ETDA integrates swarm optimization algorithms with topological data analysis (TDA) to solve high-dimensional optimization problems, particularly for health data applications.

## Overview

ETDA uses persistence homology to:
1. **Reduce large-scale multidimensional spaces** - Map high-dimensional regions of interest to lower-dimensional manifolds
2. **Identify global maxima** - Use TDA to find critical points on the topological structure
3. **Optimize with swarm intelligence** - Apply nature-inspired algorithms on the reduced space

## Key Features

- **Persistent Homology Analysis**: Uses giotto-tda to compute persistence diagrams
- **Manifold Reduction**: Reduces high-dimensional search spaces to lower-dimensional manifolds
- **Global Optima Identification**: Identifies critical points and global maxima on the topological structure
- **Swarm Optimization Integration**: Works with nio (Nature-Inspired Optimization) algorithms
- **Health Data Focus**: Designed for large-scale health data optimization problems

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from etda import ETDAOptimizer
from etda import create_high_dim_roi
import numpy as np

# Create a high-dimensional region of interest
roi_data = create_high_dim_roi(
    n_samples=1000,
    n_features=50,
    seed=42
)

# Initialize ETDA optimizer
optimizer = ETDAOptimizer(
    roi_data=roi_data,
    reduced_dim=10,
    n_components=3,
    persistence_threshold=0.1
)

# Run TDA preprocessing
optimizer.preprocess_tda()

# Optimize on reduced manifold
best_position, best_value = optimizer.optimize(
    algorithm='bat',
    iterations=200,
    population_size=40
)

print(f"Best value: {best_value}")
print(f"Best position (reduced space): {best_position}")
```

## Example: Large-Scale Multidimensional ROI

See `examples/high_dim_roi_example.py` for a complete example demonstrating:
- Large-scale multidimensional region of interest
- Persistence homology mapping
- TDA-based global maxima identification
- Swarm optimization on the reduced manifold

## Dependencies

- **giotto-tda**: Topological data analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning utilities
- **nio**: Nature-inspired optimization (swarm algorithms)

## License

MIT

