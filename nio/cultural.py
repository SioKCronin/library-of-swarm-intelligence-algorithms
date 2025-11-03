"""
Cultural Algorithm Implementation

Reference:
Reynolds, R. G. "An introduction to cultural algorithms."
Proceedings of the 3rd Annual Conference on Evolutionary Programming (1994).

Cultural Algorithms model dual inheritance (genetic and cultural evolution)
through a population space and belief space that interact via communication
protocols.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Vector = List[float]
Objective = Callable[[Vector], float]


def rastrigin(position: Sequence[float]) -> float:
    """Default benchmark function (global minimum at 0)."""
    return 10 * len(position) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in position)


@dataclass
class Individual:
    """Represents an individual in the population space."""
    position: Vector
    fitness: float


@dataclass
class NormativeKnowledge:
    """Normative knowledge component of belief space.
    
    Stores acceptable ranges and guidance for each dimension.
    """
    lower_bounds: Vector
    upper_bounds: Vector
    performance_scores: Vector  # Quality of each bound


@dataclass
class SituationalKnowledge:
    """Situational knowledge component of belief space.
    
    Stores best examples/individuals found so far.
    """
    best_individuals: List[Individual]


@dataclass
class BeliefSpace:
    """Belief space containing cultural knowledge."""
    normative: NormativeKnowledge
    situational: SituationalKnowledge


class CulturalAlgorithm:
    """Implementation of the Cultural Algorithm metaheuristic."""

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        accept_rate: float = 0.2,
        influence_rate: float = 0.3,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.7,
        num_situational: int = 5,
        seed: int | None = None,
    ) -> None:
        """Initialize the Cultural Algorithm.

        Args:
            objective: Objective function to minimize
            bounds: Search space boundaries for each dimension
            population_size: Size of the population
            accept_rate: Fraction of population that influences belief space
            influence_rate: Fraction of population influenced by belief space
            mutation_rate: Probability of mutation per dimension
            mutation_strength: Strength of mutation (relative to search space)
            crossover_rate: Probability of crossover
            num_situational: Number of best individuals to keep in situational knowledge
            seed: Random seed for reproducibility
        """
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0.0 <= accept_rate <= 1.0:
            raise ValueError("accept_rate must be between 0 and 1")
        if not 0.0 <= influence_rate <= 1.0:
            raise ValueError("influence_rate must be between 0 and 1")

        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.accept_rate = accept_rate
        self.influence_rate = influence_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_situational = num_situational
        self.random = random.Random(seed)

        self.population: List[Individual] = []
        self.belief_space: BeliefSpace | None = None
        self.best: Individual | None = None

    # ------------------------------------------------------------------
    # Initialisation
    def initialise(self) -> None:
        """Initialize population and belief space."""
        # Initialize population
        self.population = []
        for _ in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            fitness = self.objective(position)
            individual = Individual(position, fitness)
            self.population.append(individual)

        # Initialize belief space
        self._initialize_belief_space()

        # Find best individual
        self.best = min(self.population, key=lambda ind: ind.fitness)

    def _initialize_belief_space(self) -> None:
        """Initialize the belief space with initial knowledge."""
        # Normative knowledge: start with search space bounds
        lower_bounds = [lo for lo, hi in self.bounds]
        upper_bounds = [hi for lo, hi in self.bounds]
        performance_scores = [0.0] * self.dimension

        normative = NormativeKnowledge(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            performance_scores=performance_scores,
        )

        # Situational knowledge: start with empty or best initial individuals
        situational = SituationalKnowledge(best_individuals=[])

        self.belief_space = BeliefSpace(normative=normative, situational=situational)

    # ------------------------------------------------------------------
    # Acceptance function: selects individuals that influence belief space
    def _accept(self) -> List[Individual]:
        """Select individuals to influence the belief space."""
        # Sort by fitness (ascending)
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        num_accept = max(1, int(self.accept_rate * self.population_size))
        return sorted_pop[:num_accept]

    # ------------------------------------------------------------------
    # Belief space update
    def _update_belief_space(self, accepted: List[Individual]) -> None:
        """Update belief space based on accepted individuals."""
        if not accepted or self.belief_space is None:
            return

        # Update normative knowledge
        self._update_normative(accepted)

        # Update situational knowledge
        self._update_situational(accepted)

    def _update_normative(self, accepted: List[Individual]) -> None:
        """Update normative knowledge from accepted individuals."""
        if not accepted or self.belief_space is None:
            return

        normative = self.belief_space.normative

        for dim in range(self.dimension):
            # Find min/max values in this dimension from accepted individuals
            values = [ind.position[dim] for ind in accepted]
            min_val = min(values)
            max_val = max(values)

            # Update bounds if they improve performance
            # Shrink bounds towards better regions
            if min_val < normative.upper_bounds[dim]:
                # If we found better lower bound
                if min_val > normative.lower_bounds[dim]:
                    # Shrink lower bound
                    normative.lower_bounds[dim] = min_val
                    normative.performance_scores[dim] += 1.0

            if max_val > normative.lower_bounds[dim]:
                # If we found better upper bound
                if max_val < normative.upper_bounds[dim]:
                    # Shrink upper bound
                    normative.upper_bounds[dim] = max_val
                    normative.performance_scores[dim] += 1.0

            # Ensure bounds stay within search space
            lo, hi = self.bounds[dim]
            normative.lower_bounds[dim] = max(lo, normative.lower_bounds[dim])
            normative.upper_bounds[dim] = min(hi, normative.upper_bounds[dim])

    def _update_situational(self, accepted: List[Individual]) -> None:
        """Update situational knowledge with best individuals."""
        if not accepted or self.belief_space is None:
            return

        situational = self.belief_space.situational

        # Add accepted individuals to situational knowledge
        for ind in accepted:
            # Create a copy
            new_ind = Individual(position=ind.position[:], fitness=ind.fitness)
            situational.best_individuals.append(new_ind)

        # Keep only the best num_situational individuals
        situational.best_individuals.sort(key=lambda x: x.fitness)
        situational.best_individuals = situational.best_individuals[:self.num_situational]

    # ------------------------------------------------------------------
    # Influence function: uses belief space to guide evolution
    def _influence(self, individual: Individual) -> Vector:
        """Generate new position influenced by belief space."""
        if self.belief_space is None:
            return individual.position[:]

        new_position = individual.position[:]

        # Influence based on normative knowledge
        if self.random.random() < self.influence_rate:
            normative = self.belief_space.normative
            for dim in range(self.dimension):
                # Generate value within normative bounds
                lo = normative.lower_bounds[dim]
                hi = normative.upper_bounds[dim]
                if hi > lo:
                    new_position[dim] = self.random.uniform(lo, hi)
                else:
                    # If bounds collapsed, use midpoint
                    new_position[dim] = (lo + hi) / 2.0

        # Influence based on situational knowledge
        situational = self.belief_space.situational
        if situational.best_individuals and self.random.random() < 0.5:
            # Move towards a best example
            best_example = self.random.choice(situational.best_individuals)
            for dim in range(self.dimension):
                # Blend current position with best example
                alpha = self.random.uniform(0.3, 0.7)
                new_position[dim] = (
                    alpha * individual.position[dim] +
                    (1 - alpha) * best_example.position[dim]
                )

        return self._clamp_position(new_position)

    def _clamp_position(self, position: Vector) -> Vector:
        """Clamp position within bounds."""
        clamped = position[:]
        for i, (lo, hi) in enumerate(self.bounds):
            clamped[i] = max(lo, min(hi, clamped[i]))
        return clamped

    # ------------------------------------------------------------------
    # Evolution operators
    def _crossover(self, parent1: Individual, parent2: Individual) -> Vector:
        """Perform crossover between two parents."""
        if self.random.random() > self.crossover_rate:
            return parent1.position[:]

        # Uniform crossover
        child = []
        for i in range(self.dimension):
            if self.random.random() < 0.5:
                child.append(parent1.position[i])
            else:
                child.append(parent2.position[i])
        return child

    def _mutate(self, position: Vector) -> Vector:
        """Apply mutation to a position."""
        mutated = position[:]
        for i in range(self.dimension):
            if self.random.random() < self.mutation_rate:
                lo, hi = self.bounds[i]
                range_size = hi - lo
                mutation = self.random.gauss(0, self.mutation_strength * range_size)
                mutated[i] += mutation
        return self._clamp_position(mutated)

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Perform one iteration of the Cultural Algorithm."""
        if self.best is None or self.belief_space is None:
            raise RuntimeError("Call initialise() before step().")

        # Evaluate population (if not already done)
        for individual in self.population:
            individual.fitness = self.objective(individual.position)

        # Accept function: select individuals to influence belief space
        accepted = self._accept()

        # Update belief space
        self._update_belief_space(accepted)

        # Generate new population
        new_population = []
        for _ in range(self.population_size):
            # Select parents (dance party)
            parent1 = self._dance_party()
            parent2 = self._dance_party()

            # Create child through crossover
            child_position = self._crossover(parent1, parent2)

            # Apply mutation
            child_position = self._mutate(child_position)

            # Influence by belief space
            child = Individual(position=child_position, fitness=0.0)
            influenced_position = self._influence(child)
            child.position = influenced_position

            # Evaluate child
            child.fitness = self.objective(child.position)
            new_population.append(child)

        # Update population
        self.population = new_population

        # Update best
        current_best = min(self.population, key=lambda ind: ind.fitness)
        if self.best is None or current_best.fitness < self.best.fitness:
            self.best = Individual(
                position=current_best.position[:],
                fitness=current_best.fitness,
            )

    def _dance_party(self, party_size: int = 3) -> Individual:
        """Dance party selection - the best dancer wins!"""
        dance_party = self.random.sample(self.population, min(party_size, len(self.population)))
        return min(dance_party, key=lambda ind: ind.fitness)

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
    """Example usage of the Cultural Algorithm."""
    ca = CulturalAlgorithm(
        bounds=[(-5.12, 5.12)] * 5,
        population_size=40,
        accept_rate=0.2,
        influence_rate=0.3,
        seed=42,
    )
    best_position, best_value = ca.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])


if __name__ == "__main__":
    main()

