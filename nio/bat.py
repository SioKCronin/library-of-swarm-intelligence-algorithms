"""
Bat Algorithm Implementation

Reference:
X.-S. Yang, "A new metaheuristic bat-inspired algorithm", in: Nature Inspired Cooperative Strategies for Optimization (NICSO 2010).
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
class Bat:
    position: Vector
    velocity: Vector
    frequency: float
    loudness: float
    pulse_rate: float
    fitness: float


class BatAlgorithm:
    """Simple yet expressive implementation of the Bat Algorithm."""

    def __init__(
        self,
        objective: Objective = rastrigin,
        bounds: Sequence[Tuple[float, float]] = ((-5.12, 5.12),) * 2,
        population_size: int = 30,
        frequencies: Tuple[float, float] = (0.0, 2.0),
        loudness: Tuple[float, float] = (1.0, 0.9),
        pulse_rate: Tuple[float, float] = (0.5, 0.5),
        alpha: float = 0.9,
        gamma: float = 0.9,
        seed: int | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        self.objective = objective
        self.bounds = list(bounds)
        self.dimension = len(bounds)
        self.population_size = population_size
        self.fmin, self.fmax = frequencies
        self.initial_loudness, self.min_loudness = loudness
        self.initial_pulse_rate, self.max_pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.random = random.Random(seed)

        self.population: List[Bat] = []
        self.best: Bat | None = None

    # ------------------------------------------------------------------
    # Initialisation
    def initialise(self) -> None:
        self.population = []
        for _ in range(self.population_size):
            position = [self.random.uniform(lo, hi) for lo, hi in self.bounds]
            velocity = [0.0] * self.dimension
            frequency = self.random.uniform(self.fmin, self.fmax)
            loudness = self.initial_loudness
            pulse_rate = self.initial_pulse_rate
            fitness = self.objective(position)
            bat = Bat(position, velocity, frequency, loudness, pulse_rate, fitness)
            self.population.append(bat)

        self.best = min(self.population, key=lambda b: b.fitness)

    # ------------------------------------------------------------------
    def _update_frequency(self, bat: Bat) -> None:
        bat.frequency = self.fmin + (self.fmax - self.fmin) * self.random.random()

    def _update_velocity(self, bat: Bat, best_position: Sequence[float]) -> None:
        for i in range(self.dimension):
            bat.velocity[i] += (bat.position[i] - best_position[i]) * bat.frequency

    def _move(self, bat: Bat) -> None:
        for i, (lo, hi) in enumerate(self.bounds):
            bat.position[i] += bat.velocity[i]
            # Clamp within bounds
            if bat.position[i] < lo:
                bat.position[i] = lo
                bat.velocity[i] = 0
            elif bat.position[i] > hi:
                bat.position[i] = hi
                bat.velocity[i] = 0

    def _local_search(self, bat: Bat, best_position: Sequence[float]) -> Vector:
        epsilon = 1e-6
        scale = sum(abs(x) for x in best_position) / self.dimension + epsilon
        return [best_position[i] + (self.random.random() - 0.5) * scale for i in range(self.dimension)]

    # ------------------------------------------------------------------
    def step(self) -> None:
        if self.best is None:
            raise RuntimeError("Call initialise() before step().")

        best_position = self.best.position

        for bat in self.population:
            self._update_frequency(bat)
            self._update_velocity(bat, best_position)
            self._move(bat)

            if self.random.random() > bat.pulse_rate:
                candidate_position = self._local_search(bat, best_position)
            else:
                candidate_position = bat.position[:]

            candidate_fitness = self.objective(candidate_position)

            if (candidate_fitness <= bat.fitness and
                    self.random.random() < bat.loudness):
                bat.position = candidate_position
                bat.fitness = candidate_fitness
                bat.loudness *= self.alpha
                bat.pulse_rate = bat.pulse_rate * (1 - self.gamma) + self.gamma * self.max_pulse_rate

            if self.best is None or bat.fitness < self.best.fitness:
                self.best = Bat(
                    position=bat.position[:],
                    velocity=bat.velocity[:],
                    frequency=bat.frequency,
                    loudness=bat.loudness,
                    pulse_rate=bat.pulse_rate,
                    fitness=bat.fitness,
                )

        if self.best:
            self.best.loudness = max(self.best.loudness, self.min_loudness)

    # ------------------------------------------------------------------
    def run(self, iterations: int = 100) -> Tuple[Vector, float]:
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
    ba = BatAlgorithm(bounds=[(-5.12, 5.12)] * 5, population_size=40, seed=42)
    best_position, best_value = ba.run(iterations=200)
    print("Best value:", best_value)
    print("Best position:", [round(x, 4) for x in best_position])


if __name__ == "__main__":
    main()
