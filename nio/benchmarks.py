"""
Benchmark functions for testing optimization algorithms.

Includes dynamic benchmarks that change over time, such as the
contracting optimum problem.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional, Sequence, Tuple

Vector = Sequence[float]
Objective = Callable[[Vector], float]


def rastrigin(position: Sequence[float]) -> float:
    """Classic Rastrigin function (global minimum at 0).

    Reference:
        Rastrigin, L. A. "Systems of extremal control."
        Nauka, Moscow (1974).
    """
    return 10 * len(position) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in position)


class ContractingOptimum:
    """
    Dynamic benchmark where the optimum region contracts and moves
    as optimization progresses.

    As agents explore the search space and approach boundaries, the
    optimal region shrinks and the optimum point moves within the
    contracting space. This tests an algorithm's ability to adapt
    to changing conditions.

    The contraction happens based on iteration count, and the optimum
    moves in a spiral or random walk pattern within the shrinking space.

    Attributes:
        dimension: Dimensionality of the search space
        bounds: Search space bounds
        initial_radius: Initial radius of the optimal region
        final_radius: Final radius of the optimal region
        contraction_rate: How quickly the region contracts
        movement_speed: How fast the optimum moves
        iteration: Current iteration (for tracking progress)
        optimum_position: Current position of the optimum
        current_radius: Current radius of the optimal region
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        initial_radius: float = 2.0,
        final_radius: float = 0.1,
        contraction_rate: float = 0.95,
        movement_speed: float = 0.05,
        max_iterations: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the contracting optimum benchmark.

        Args:
            bounds: Search space boundaries for each dimension
            initial_radius: Starting radius of the optimal region
            final_radius: Minimum radius the region will contract to
            contraction_rate: Multiplier for radius each iteration (0-1)
            movement_speed: Speed of optimum movement relative to space size
            max_iterations: Maximum expected iterations (for scaling)
            seed: Random seed for reproducibility
        """
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.initial_radius = initial_radius
        self.final_radius = final_radius
        self.contraction_rate = contraction_rate
        self.movement_speed = movement_speed
        self.max_iterations = max_iterations
        self.random = random.Random(seed)

        # Compute search space center and size for movement scaling
        self.space_center = [
            (lo + hi) / 2.0 for lo, hi in self.bounds
        ]
        self.space_size = [
            hi - lo for lo, hi in self.bounds
        ]

        # Initialize state
        self.iteration = 0
        self.current_radius = self.initial_radius
        
        # Start optimum at a random position within bounds
        self.optimum_position = [
            self.random.uniform(lo, hi) for lo, hi in self.bounds
        ]
        
        # Movement direction (random walk)
        self.movement_direction = [
            self.random.uniform(-1, 1) for _ in range(self.dimension)
        ]
        # Normalize direction
        direction_magnitude = math.sqrt(sum(d * d for d in self.movement_direction))
        if direction_magnitude > 0:
            self.movement_direction = [
                d / direction_magnitude for d in self.movement_direction
            ]

    def update(self, iteration: Optional[int] = None) -> None:
        """Update the benchmark state (optimum position and radius).

        Should be called at the start of each optimization iteration
        to update the dynamic aspects of the benchmark.

        Args:
            iteration: Current iteration number. If None, uses internal counter.
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1

        # Contract the radius
        progress = min(self.iteration / self.max_iterations, 1.0)
        self.current_radius = (
            self.initial_radius * (self.contraction_rate ** self.iteration)
        )
        self.current_radius = max(self.current_radius, self.final_radius)

        # Move the optimum within the contracting space
        # Constrain movement to keep optimum within shrinking region
        max_movement = self.current_radius * self.movement_speed
        
        # Add some randomness to direction (random walk)
        if self.random.random() < 0.3:  # 30% chance to change direction
            self.movement_direction = [
                self.random.uniform(-1, 1) for _ in range(self.dimension)
            ]
            direction_magnitude = math.sqrt(sum(d * d for d in self.movement_direction))
            if direction_magnitude > 0:
                self.movement_direction = [
                    d / direction_magnitude for d in self.movement_direction
                ]

        # Update optimum position
        for i in range(self.dimension):
            step = max_movement * self.movement_direction[i]
            new_pos = self.optimum_position[i] + step
            
            # Keep optimum within bounds
            lo, hi = self.bounds[i]
            if new_pos < lo:
                new_pos = lo
                self.movement_direction[i] *= -1  # Bounce off boundary
            elif new_pos > hi:
                new_pos = hi
                self.movement_direction[i] *= -1  # Bounce off boundary
            
            self.optimum_position[i] = new_pos

    def __call__(self, position: Vector) -> float:
        """Evaluate fitness at a given position.

        The fitness is the distance to the current optimum position,
        with a smooth basin around the optimum defined by the current radius.

        Args:
            position: Point in search space to evaluate

        Returns:
            Fitness value (distance to optimum, lower is better)
        """
        # Compute distance to current optimum
        distance_squared = sum(
            (position[i] - self.optimum_position[i]) ** 2
            for i in range(self.dimension)
        )
        distance = math.sqrt(distance_squared)

        # Smooth fitness landscape: quadratic within radius, linear outside
        if distance <= self.current_radius:
            # Inside optimal region: quadratic penalty
            return (distance / self.current_radius) ** 2
        else:
            # Outside optimal region: distance-based penalty
            excess = distance - self.current_radius
            return 1.0 + excess / max(self.space_size)

    def reset(self) -> None:
        """Reset the benchmark to initial state."""
        self.iteration = 0
        self.current_radius = self.initial_radius
        self.optimum_position = [
            self.random.uniform(lo, hi) for lo, hi in self.bounds
        ]
        self.movement_direction = [
            self.random.uniform(-1, 1) for _ in range(self.dimension)
        ]
        direction_magnitude = math.sqrt(sum(d * d for d in self.movement_direction))
        if direction_magnitude > 0:
            self.movement_direction = [
                d / direction_magnitude for d in self.movement_direction
            ]

    def get_optimum_info(self) -> Tuple[Vector, float]:
        """Get current optimum position and radius.

        Returns:
            Tuple of (optimum_position, current_radius)
        """
        return (list(self.optimum_position), self.current_radius)


def contracting_optimum(
    bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
    **kwargs
) -> ContractingOptimum:
    """Factory function to create a ContractingOptimum benchmark.

    Args:
        bounds: Search space boundaries
        **kwargs: Additional arguments passed to ContractingOptimum

    Returns:
        Configured ContractingOptimum instance
    """
    return ContractingOptimum(bounds=bounds, **kwargs)

