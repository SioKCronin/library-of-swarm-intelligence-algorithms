"""
Example: ETDA for Large-Scale Multidimensional Region of Interest

This example demonstrates:
1. Creating a large-scale multidimensional region of interest
2. Using persistence homology to map the topological structure
3. Reducing the high-dimensional space to a lower-dimensional manifold
4. Identifying global maxima using TDA
5. Running swarm optimization on the reduced space

This is particularly useful for health data applications where we have
high-dimensional feature spaces and need to find optimal configurations.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import etda
sys.path.insert(0, str(Path(__file__).parent.parent))

from etda import ETDAOptimizer, create_high_dim_roi


def complex_health_objective(x: np.ndarray) -> float:
    """
    Simulated complex health data objective function.
    
    This represents a complex, multimodal fitness landscape that might
    arise from health data analysis (e.g., treatment optimization,
    biomarker discovery, etc.).
    
    Args:
        x: Input vector (can be high-dimensional)
        
    Returns:
        Objective value (higher is better for maximization)
    """
    # Multi-modal function with several peaks
    n = len(x)
    
    # Base: Rastrigin-like structure with multiple peaks
    rastrigin_component = 10 * n + np.sum(
        x**2 - 10 * np.cos(2 * np.pi * x)
    )
    
    # Add some Gaussian peaks at specific locations
    peaks = [
        np.array([0.5] * n),
        np.array([-0.3] * n),
        np.array([0.8, -0.2] + [0.0] * (n - 2)),
    ]
    
    peak_values = []
    for peak in peaks:
        if len(peak) == n:
            dist = np.linalg.norm(x - peak)
            peak_values.append(50 * np.exp(-5 * dist**2))
    
    # Combine: negative Rastrigin (minimization) + peaks (maximization)
    # We'll maximize, so return negative of Rastrigin + peaks
    return -rastrigin_component + max(peak_values) if peak_values else -rastrigin_component


def main():
    """Main example demonstrating ETDA workflow."""
    
    print("=" * 70)
    print("ETDA: Evolutionary Topological Data Analysis Example")
    print("=" * 70)
    print()
    
    # Step 1: Create large-scale multidimensional region of interest
    print("Step 1: Creating high-dimensional region of interest...")
    print("-" * 70)
    
    roi = create_high_dim_roi(
        n_samples=1000,
        n_features=50,  # High-dimensional space
        bounds=[(-1.0, 1.0)] * 50,
        distribution="multimodal",  # Multiple modes for complexity
        correlation=0.3,  # Some correlation between features
        seed=42,
    )
    
    print(f"✓ Created ROI with {roi.n_samples} samples and {roi.n_features} features")
    print(f"  Data shape: {roi.data.shape}")
    print(f"  Data range: [{roi.data.min():.3f}, {roi.data.max():.3f}]")
    print()
    
    # Step 2: Initialize ETDA optimizer
    print("Step 2: Initializing ETDA optimizer...")
    print("-" * 70)
    
    optimizer = ETDAOptimizer(
        roi_data=roi.data,
        reduced_dim=10,  # Reduce from 50D to 10D
        n_components=3,
        persistence_threshold=0.1,
        reduction_method="pca",  # Use PCA for dimensionality reduction
        random_state=42,
    )
    
    print("✓ ETDA optimizer initialized")
    print(f"  Original dimension: {roi.n_features}")
    print(f"  Reduced dimension: {optimizer.reduced_dim}")
    print()
    
    # Step 3: Preprocess with TDA
    print("Step 3: Computing persistence homology...")
    print("-" * 70)
    
    optimizer.preprocess_tda()
    
    # Get persistence information
    persistence_info = optimizer.get_persistence_info()
    
    print("✓ Persistence homology computed")
    print(f"  Persistence entropy: {persistence_info['persistence_entropy']:.4f}")
    print(f"  Number of critical points: {persistence_info['n_features']}")
    print(f"  Top 5 critical points:")
    for dim, persistence in persistence_info['critical_points'][:5]:
        print(f"    Dimension {dim}: persistence = {persistence:.4f}")
    print()
    
    # Step 4: Identify global maxima using TDA
    print("Step 4: Identifying global maxima using TDA structure...")
    print("-" * 70)
    
    # Map back to original space for objective evaluation
    def objective_in_original_space(x_reduced: np.ndarray) -> float:
        """Evaluate objective in original space via inverse transform."""
        x_original = optimizer.manifold_reducer.inverse_transform(
            x_reduced.reshape(1, -1)
        )[0]
        return complex_health_objective(x_original)
    
    # Get candidate maxima from TDA
    candidate_maxima = optimizer.tda_processor.identify_global_maxima(
        optimizer.reduced_data,
        objective_in_original_space,
        n_candidates=5,
    )
    
    print("✓ Identified candidate global maxima:")
    for i, (pos, value) in enumerate(candidate_maxima, 1):
        print(f"  Candidate {i}: value = {value:.4f}")
        print(f"    Position (reduced): {pos[:5]}... (showing first 5 dims)")
    print()
    
    # Step 5: Run swarm optimization on reduced manifold
    print("Step 5: Running swarm optimization on reduced manifold...")
    print("-" * 70)
    
    # Create objective wrapper for reduced space
    wrapped_objective = optimizer.create_objective_wrapper(
        complex_health_objective
    )
    
    # Run optimization
    best_position_reduced, best_value = optimizer.optimize(
        objective=complex_health_objective,  # Will be wrapped automatically
        algorithm="bat",
        iterations=200,
        population_size=40,
    )
    
    # Map back to original space
    best_position_original = optimizer.manifold_reducer.inverse_transform(
        best_position_reduced.reshape(1, -1)
    )[0]
    
    print("✓ Optimization complete")
    print(f"  Best value: {best_value:.4f}")
    print(f"  Best position (reduced space, first 5 dims): {best_position_reduced[:5]}")
    print(f"  Best position (original space, first 5 dims): {best_position_original[:5]}")
    print()
    
    # Step 6: Compare with TDA candidates
    print("Step 6: Comparison with TDA-identified maxima...")
    print("-" * 70)
    
    best_tda_candidate = candidate_maxima[0][1] if candidate_maxima else None
    
    print(f"  Best TDA candidate value: {best_tda_candidate:.4f}" if best_tda_candidate else "  No TDA candidates")
    print(f"  Best swarm optimization value: {best_value:.4f}")
    
    if best_tda_candidate:
        improvement = best_value - best_tda_candidate
        print(f"  Improvement: {improvement:+.4f}")
    
    print()
    print("=" * 70)
    print("ETDA workflow complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

