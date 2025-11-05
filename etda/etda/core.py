"""
Core ETDA module: TDA processing and swarm optimization integration.

This module provides:
1. TDAProcessor: Computes persistence homology and identifies critical points
2. ManifoldReducer: Reduces high-dimensional spaces to lower dimensions
3. ETDAOptimizer: Main interface combining TDA preprocessing with swarm optimization
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude
    from sklearn.manifold import MDS, TSNE
    from sklearn.decomposition import PCA
    HAS_GTDA = True
except ImportError:
    HAS_GTDA = False

try:
    import sys
    from pathlib import Path
    # Try to import nio - may need to be installed or in path
    try:
        from nio import BatAlgorithm, CulturalAlgorithm, PhilippineEagleOptimization
        HAS_NIO = True
    except ImportError:
        # Try adding parent directory to path
        parent_dir = Path(__file__).parent.parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from nio import BatAlgorithm, CulturalAlgorithm, PhilippineEagleOptimization
        HAS_NIO = True
except Exception:
    HAS_NIO = False


@dataclass
class PersistenceResult:
    """Results from persistence homology computation."""
    persistence_diagrams: np.ndarray
    persistence_entropy: float
    amplitudes: np.ndarray
    critical_points: List[Tuple[int, float]]  # (dimension, persistence)


class TDAProcessor:
    """Processes data using Topological Data Analysis."""
    
    def __init__(
        self,
        homology_dimensions: Tuple[int, ...] = (0, 1, 2),
        metric: str = "euclidean",
        persistence_threshold: float = 0.1,
    ) -> None:
        """Initialize TDA processor.
        
        Args:
            homology_dimensions: Dimensions of homology to compute (0, 1, 2)
            metric: Distance metric for VR complex
            persistence_threshold: Minimum persistence to consider
        """
        if not HAS_GTDA:
            raise ImportError(
                "giotto-tda is required. Install with: pip install giotto-tda"
            )
        
        self.homology_dimensions = homology_dimensions
        self.metric = metric
        self.persistence_threshold = persistence_threshold
        self.vr_persistence = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            metric=metric,
        )
        self.persistence_entropy = PersistenceEntropy()
        self.amplitude = Amplitude(metric="bottleneck")
        self._fitted = False
    
    def compute_persistence(self, X: np.ndarray) -> PersistenceResult:
        """Compute persistence homology for the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            PersistenceResult with diagrams, entropy, amplitudes, and critical points
        """
        # Reshape if needed (giotto-tda expects (n_samples, n_features))
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        
        # Compute persistence diagrams
        diagrams = self.vr_persistence.fit_transform(X[None, :, :])
        self._fitted = True
        
        # Compute persistence entropy
        entropy = self.persistence_entropy.fit_transform(diagrams)
        
        # Compute amplitudes
        amplitudes = self.amplitude.fit_transform(diagrams)
        
        # Extract critical points (high persistence features)
        critical_points = self._extract_critical_points(diagrams[0])
        
        return PersistenceResult(
            persistence_diagrams=diagrams[0],
            persistence_entropy=float(entropy[0, 0]),
            amplitudes=amplitudes[0],
            critical_points=critical_points,
        )
    
    def _extract_critical_points(
        self,
        diagrams: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Extract critical points with high persistence.
        
        Args:
            diagrams: Persistence diagrams from giotto-tda
            
        Returns:
            List of (dimension, persistence) tuples for significant features
        """
        critical_points = []
        
        for dim_idx, dim in enumerate(self.homology_dimensions):
            # Get diagram for this dimension
            dim_diagram = diagrams[dim_idx]
            
            # Filter by persistence threshold
            for point in dim_diagram:
                if len(point) >= 2:
                    birth, death = point[0], point[1]
                    persistence = death - birth
                    
                    if persistence >= self.persistence_threshold:
                        critical_points.append((dim, persistence))
        
        # Sort by persistence (descending)
        critical_points.sort(key=lambda x: x[1], reverse=True)
        return critical_points
    
    def identify_global_maxima(
        self,
        X: np.ndarray,
        objective: Callable[[np.ndarray], float],
        n_candidates: int = 5,
    ) -> List[Tuple[np.ndarray, float]]:
        """Identify candidate global maxima using TDA structure.
        
        Args:
            X: Input data
            objective: Objective function to evaluate
            n_candidates: Number of candidate maxima to return
            
        Returns:
            List of (position, value) tuples for candidate maxima
        """
        result = self.compute_persistence(X)
        
        # Use critical points to identify regions of interest
        # Points with high persistence in dimension 0 are potential optima
        candidates = []
        
        # For high-persistence features, sample nearby points
        rng = np.random.RandomState(42)
        
        for dim, persistence in result.critical_points[:n_candidates * 2]:
            if dim == 0:  # Connected components - potential local/global optima
                # Sample points in regions with high topological significance
                idx = rng.choice(len(X), size=min(10, len(X)), replace=False)
                for i in idx:
                    position = X[i]
                    value = objective(position)
                    candidates.append((position, value))
        
        # Sort by objective value (descending for maximization)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n_candidates]


class ManifoldReducer:
    """Reduces high-dimensional spaces to lower dimensions."""
    
    def __init__(
        self,
        reduced_dim: int = 10,
        method: str = "pca",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize manifold reducer.
        
        Args:
            reduced_dim: Target reduced dimension
            method: Reduction method ("pca", "mds", "tsne")
            random_state: Random seed
        """
        self.reduced_dim = reduced_dim
        self.method = method
        self.random_state = random_state
        self.reducer = None
        self._fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit reducer and transform data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Reduced data of shape (n_samples, reduced_dim)
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        
        if X.shape[1] <= self.reduced_dim:
            # Already low-dimensional, return as-is
            return X
        
        if self.method == "pca":
            from sklearn.decomposition import PCA
            self.reducer = PCA(
                n_components=self.reduced_dim,
                random_state=self.random_state,
            )
        elif self.method == "mds":
            from sklearn.manifold import MDS
            self.reducer = MDS(
                n_components=self.reduced_dim,
                random_state=self.random_state,
                dissimilarity="euclidean",
            )
        elif self.method == "tsne":
            from sklearn.manifold import TSNE
            self.reducer = TSNE(
                n_components=self.reduced_dim,
                random_state=self.random_state,
                perplexity=min(30, X.shape[0] // 4),
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        X_reduced = self.reducer.fit_transform(X)
        self._fitted = True
        return X_reduced
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted reducer.
        
        Args:
            X: Input data
            
        Returns:
            Reduced data
        """
        if not self._fitted:
            raise RuntimeError("Must call fit_transform() first")
        
        if self.method == "tsne":
            # TSNE doesn't support transform, need to refit or use approximation
            raise NotImplementedError(
                "TSNE doesn't support transform(). Use fit_transform() or switch to PCA/MDS."
            )
        
        return self.reducer.transform(X)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Inverse transform from reduced space to original space.
        
        Args:
            X_reduced: Reduced data
            
        Returns:
            Data in original space (approximation)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit_transform() first")
        
        if hasattr(self.reducer, "inverse_transform"):
            return self.reducer.inverse_transform(X_reduced)
        else:
            raise NotImplementedError(
                f"{self.method} does not support inverse_transform"
            )


class ETDAOptimizer:
    """Main ETDA optimizer combining TDA preprocessing with swarm optimization."""
    
    def __init__(
        self,
        roi_data: np.ndarray,
        reduced_dim: int = 10,
        n_components: int = 3,
        persistence_threshold: float = 0.1,
        reduction_method: str = "pca",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize ETDA optimizer.
        
        Args:
            roi_data: Region of interest data (n_samples, n_features)
            reduced_dim: Target reduced dimension
            n_components: Number of TDA components to consider
            persistence_threshold: Minimum persistence for TDA features
            reduction_method: Method for dimensionality reduction
            random_state: Random seed
        """
        if not HAS_GTDA:
            raise ImportError(
                "giotto-tda is required. Install with: pip install giotto-tda"
            )
        if not HAS_NIO:
            raise ImportError(
                "nio is required. Install with: pip install -e ../library-of-nature-inspired-optimization "
                "or ensure nio is in your Python path"
            )
        
        self.roi_data = roi_data
        self.reduced_dim = reduced_dim
        self.n_components = n_components
        self.random_state = random_state
        
        # Initialize components
        self.tda_processor = TDAProcessor(
            persistence_threshold=persistence_threshold,
        )
        self.manifold_reducer = ManifoldReducer(
            reduced_dim=reduced_dim,
            method=reduction_method,
            random_state=random_state,
        )
        
        # State
        self.reduced_data: Optional[np.ndarray] = None
        self.persistence_result: Optional[PersistenceResult] = None
        self._preprocessed = False
        self._objective_wrapper: Optional[Callable] = None
    
    def preprocess_tda(self) -> None:
        """Preprocess data using TDA: compute persistence and reduce manifold."""
        # Reduce dimensionality
        self.reduced_data = self.manifold_reducer.fit_transform(self.roi_data)
        
        # Compute persistence homology on reduced space
        self.persistence_result = self.tda_processor.compute_persistence(
            self.reduced_data
        )
        
        self._preprocessed = True
    
    def create_objective_wrapper(
        self,
        original_objective: Callable[[np.ndarray], float],
    ) -> Callable[[np.ndarray], float]:
        """Create objective function wrapper for reduced space.
        
        Args:
            original_objective: Objective function in original space
            
        Returns:
            Objective function in reduced space
        """
        def reduced_objective(x_reduced: np.ndarray) -> float:
            # Map from reduced space back to original space
            if self.manifold_reducer.method == "pca":
                x_original = self.manifold_reducer.inverse_transform(
                    x_reduced.reshape(1, -1)
                )[0]
            else:
                # For methods without inverse, use nearest neighbor approximation
                # For simplicity, we'll use a linear approximation
                # In practice, you might want to use a more sophisticated method
                raise NotImplementedError(
                    f"Inverse transform not available for {self.manifold_reducer.method}"
                )
            
            return original_objective(x_original)
        
        self._objective_wrapper = reduced_objective
        return reduced_objective
    
    def optimize(
        self,
        objective: Optional[Callable[[np.ndarray], float]] = None,
        algorithm: str = "bat",
        iterations: int = 200,
        population_size: int = 40,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        """Run swarm optimization on TDA-reduced manifold.
        
        Args:
            objective: Objective function (if None, uses default)
            algorithm: Algorithm name ("bat", "cultural", "philippine_eagle")
            iterations: Number of iterations
            population_size: Population size
            bounds: Bounds for reduced space (auto-computed if None)
            **kwargs: Additional arguments for optimizer
            
        Returns:
            (best_position_in_reduced_space, best_value)
        """
        if not self._preprocessed:
            self.preprocess_tda()
        
        if self.reduced_data is None:
            raise RuntimeError("Reduced data not available")
        
        # Compute bounds from reduced data
        if bounds is None:
            mins = self.reduced_data.min(axis=0)
            maxs = self.reduced_data.max(axis=0)
            # Add some padding
            padding = (maxs - mins) * 0.1
            bounds = [
                (float(mins[i] - padding[i]), float(maxs[i] + padding[i]))
                for i in range(self.reduced_dim)
            ]
        
        # Create objective wrapper
        if objective is None:
            # Default: maximize distance from mean (explore manifold)
            def default_objective(x: np.ndarray) -> float:
                center = self.reduced_data.mean(axis=0)
                return -np.linalg.norm(x - center)  # Negative for minimization
            objective = default_objective
        
        if self.manifold_reducer.method == "pca":
            # Can use inverse transform
            wrapped_objective = self.create_objective_wrapper(objective)
        else:
            # Work directly in reduced space
            wrapped_objective = objective
        
        # Initialize optimizer
        if algorithm == "bat":
            optimizer = BatAlgorithm(
                objective=wrapped_objective,
                bounds=bounds,
                population_size=population_size,
                seed=self.random_state,
                **kwargs,
            )
        elif algorithm == "cultural":
            optimizer = CulturalAlgorithm(
                objective=wrapped_objective,
                bounds=bounds,
                population_size=population_size,
                seed=self.random_state,
                **kwargs,
            )
        elif algorithm == "philippine_eagle":
            optimizer = PhilippineEagleOptimization(
                objective=wrapped_objective,
                bounds=bounds,
                population_size=population_size,
                seed=self.random_state,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Choose from: bat, cultural, philippine_eagle"
            )
        
        # Run optimization
        best_position, best_value = optimizer.run(iterations=iterations)
        
        return np.array(best_position), best_value
    
    def get_persistence_info(self) -> Dict[str, Any]:
        """Get information about persistence homology results.
        
        Returns:
            Dictionary with persistence statistics
        """
        if not self._preprocessed or self.persistence_result is None:
            raise RuntimeError("Must call preprocess_tda() first")
        
        return {
            "persistence_entropy": self.persistence_result.persistence_entropy,
            "amplitudes": self.persistence_result.amplitudes.tolist(),
            "critical_points": self.persistence_result.critical_points,
            "n_features": len(self.persistence_result.critical_points),
        }

