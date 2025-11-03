"""
Philippine Eagle Optimization Algorithm Implementation

The algorithm divides agents into sub-populations, assigns operators,
and alternates between local optimization and exploration phases.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Sequence, Tuple

Vector = List[float]
Objective = Callable[[Vector], float]


def rastrigin(position: Sequence[float]) -> float:
    """Default benchmark function (global minimum at 0)."""
    return 10 * len(position) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in position)


class Phase(Enum):
    """Optimization phases."""
    LOCAL = "local"
    EXPLORATION = "exploration"


class Operator(Enum):
    """Optimization operators."""
    LEVY_FLIGHT = "levy_flight"
    GAUSSIAN_WALK = "gaussian_walk"
    PERTURBATION = "perturbation"
    CROSSOVER = "crossover"


@dataclass
class Eagle:
    """Represents an eagle in the population."""
    position: Vector
    fitness: float
    sub_population_id: int
    operator: Operator


class PhilippineEagleOptimization:
    """Implementation of the Philippine Eagle Optimization Algorithm."""

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        num_sub_populations: int = 3,
        local_phase_probability: float = 0.5,
        levy_step_size: float = 0.01,
        gaussian_sigma: float = 0.1,
        perturbation_radius: float = 0.2,
        seed: int | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if num_sub_populations <= 0 or num_sub_populations > population_size:
            raise ValueError("num_sub_populations must be positive and <= population_size")
        if not 0.0 <= local_phase_probability <= 1.0:
            raise ValueError("local_phase_probability must be between 0 and 1")

        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.num_sub_populations = num_sub_populations
        self.local_phase_probability = local_phase_probability
        self.levy_step_size = levy_step_size
        self.gaussian_sigma = gaussian_sigma
        self.perturbation_radius = perturbation_radius
        self.random = random.Random(seed)

        self.population: List[Eagle] = []
        self.sub_populations: List[List[Eagle]] = []
        self.best: Eagle | None = None
        self.current_phase: Phase = Phase.EXPLORATION

    # ------------------------------------------------------------------
    # Initialisation
    def initialise(self) -> None:
        """Initialize the population and divide into sub-populations."""
        self.population = []
        
        # Create eagles with random positions
        for i in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            fitness = self.objective(position)
            sub_pop_id = i % self.num_sub_populations
            # Assign operators to sub-populations
            operator = self._assign_operator(sub_pop_id)
            eagle = Eagle(position, fitness, sub_pop_id, operator)
            self.population.append(eagle)

        # Divide into sub-populations
        self._update_sub_populations()
        
        # Find best eagle
        self.best = min(self.population, key=lambda e: e.fitness)

    def _assign_operator(self, sub_population_id: int) -> Operator:
        """Assign operators to sub-populations in a round-robin fashion."""
        operators = list(Operator)
        return operators[sub_population_id % len(operators)]

    def _update_sub_populations(self) -> None:
        """Update sub-population groupings."""
        self.sub_populations = [[] for _ in range(self.num_sub_populations)]
        for eagle in self.population:
            self.sub_populations[eagle.sub_population_id].append(eagle)

    # ------------------------------------------------------------------
    # Phase management
    def _determine_phase(self) -> Phase:
        """Determine the current optimization phase."""
        if self.random.random() < self.local_phase_probability:
            return Phase.LOCAL
        return Phase.EXPLORATION

    # ------------------------------------------------------------------
    # Operators
    def _levy_flight(self, position: Vector, best_position: Vector) -> Vector:
        """Levy flight operator for exploration."""
        new_position = position[:]
        for i in range(self.dimension):
            # Generate Levy flight step
            beta = 1.5  # Levy distribution parameter
            u = self.random.gauss(0, 1)
            v = self.random.gauss(0, 1)
            sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                      (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            levy_step = u / (abs(v) ** (1 / beta)) * sigma_u
            
            step = self.levy_step_size * levy_step * (position[i] - best_position[i])
            new_position[i] = position[i] + step
        return self._clamp_position(new_position)

    def _gaussian_walk(self, position: Vector, best_position: Vector) -> Vector:
        """Gaussian walk operator for local search."""
        new_position = position[:]
        for i in range(self.dimension):
            # Walk towards best position with Gaussian noise
            direction = best_position[i] - position[i]
            step = direction + self.random.gauss(0, self.gaussian_sigma)
            new_position[i] = position[i] + step
        return self._clamp_position(new_position)

    def _perturbation(self, position: Vector, best_position: Vector) -> Vector:
        """Perturbation operator for exploration."""
        new_position = position[:]
        for i in range(self.dimension):
            # Random perturbation around current position
            perturbation = self.random.uniform(-self.perturbation_radius, self.perturbation_radius)
            new_position[i] = position[i] + perturbation * (best_position[i] - position[i])
        return self._clamp_position(new_position)

    def _crossover(self, position: Vector, best_position: Vector, other_eagle: Eagle) -> Vector:
        """Crossover operator between current and another eagle."""
        new_position = position[:]
        crossover_point = self.random.randint(0, self.dimension - 1)
        for i in range(self.dimension):
            if i <= crossover_point:
                # Take from best position or other eagle based on fitness
                if self.random.random() < 0.5:
                    new_position[i] = best_position[i]
                else:
                    new_position[i] = other_eagle.position[i]
            else:
                new_position[i] = position[i]
        return self._clamp_position(new_position)

    def _clamp_position(self, position: Vector) -> Vector:
        """Clamp position within bounds."""
        clamped = position[:]
        for i, (lo, hi) in enumerate(self.bounds):
            clamped[i] = max(lo, min(hi, clamped[i]))
        return clamped

    # ------------------------------------------------------------------
    # Phase-specific update methods
    def _local_optimization(self, eagle: Eagle) -> None:
        """Local optimization phase update."""
        best_position = self.best.position if self.best else eagle.position
        
        if eagle.operator == Operator.LEVY_FLIGHT:
            new_position = self._gaussian_walk(eagle.position, best_position)
        elif eagle.operator == Operator.GAUSSIAN_WALK:
            new_position = self._gaussian_walk(eagle.position, best_position)
        elif eagle.operator == Operator.PERTURBATION:
            new_position = self._gaussian_walk(eagle.position, best_position)
        else:  # CROSSOVER
            # For local phase, use Gaussian walk instead of crossover
            new_position = self._gaussian_walk(eagle.position, best_position)
        
        new_fitness = self.objective(new_position)
        if new_fitness <= eagle.fitness:
            eagle.position = new_position
            eagle.fitness = new_fitness

    def _exploration(self, eagle: Eagle) -> None:
        """Exploration phase update."""
        best_position = self.best.position if self.best else eagle.position
        
        if eagle.operator == Operator.LEVY_FLIGHT:
            new_position = self._levy_flight(eagle.position, best_position)
        elif eagle.operator == Operator.GAUSSIAN_WALK:
            new_position = self._levy_flight(eagle.position, best_position)
        elif eagle.operator == Operator.PERTURBATION:
            new_position = self._perturbation(eagle.position, best_position)
        else:  # CROSSOVER
            # Select random eagle from same sub-population for crossover
            sub_pop = self.sub_populations[eagle.sub_population_id]
            other_eagles = [e for e in sub_pop if e != eagle]
            if len(other_eagles) > 0:
                other_eagle = self.random.choice(other_eagles)
                new_position = self._crossover(eagle.position, best_position, other_eagle)
            else:
                # Fall back to levy flight if no other eagles in sub-population
                new_position = self._levy_flight(eagle.position, best_position)
        
        new_fitness = self.objective(new_position)
        if new_fitness <= eagle.fitness:
            eagle.position = new_position
            eagle.fitness = new_fitness

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Perform one optimization step."""
        if self.best is None:
            raise RuntimeError("Call initialise() before step().")

        # Determine phase for this iteration
        self.current_phase = self._determine_phase()

        # Update each eagle based on current phase
        for eagle in self.population:
            if self.current_phase == Phase.LOCAL:
                self._local_optimization(eagle)
            else:  # EXPLORATION
                self._exploration(eagle)

            # Update best solution
            if eagle.fitness < self.best.fitness:
                self.best = Eagle(
                    position=eagle.position[:],
                    fitness=eagle.fitness,
                    sub_population_id=eagle.sub_population_id,
                    operator=eagle.operator,
                )

        # Update sub-population groupings (in case of dynamic reassignment)
        self._update_sub_populations()

    # ------------------------------------------------------------------
    def run(self, iterations: int = 100) -> Tuple[Vector, float]:
        """Run the optimization algorithm for specified iterations."""
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        self.initialise()
        for iteration in range(iterations):
            # Update stateful objectives (e.g., contracting optimum)
            if hasattr(self.objective, 'update'):
                self.objective.update(iteration)
            self.step()
        
        if self.best is None:
            raise RuntimeError("Algorithm did not initialise best solution")
        return self.best.position[:], self.best.fitness


def main() -> None:
    """Example usage of the Philippine Eagle Optimization algorithm."""
    peo = PhilippineEagleOptimization(
        bounds=[(-5.12, 5.12)] * 5,
        population_size=40,
        num_sub_populations=4,
        seed=42,
    )
    best_position, best_value = peo.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])


if __name__ == "__main__":
    main()

