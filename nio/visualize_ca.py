"""
Visualization for Cultural Algorithm

Creates animated visualizations showing:
- Population space (individuals)
- Belief space (normative bounds, situational knowledge)
- Evolution over iterations
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .cultural import CulturalAlgorithm, Individual, BeliefSpace
from .benchmarks import rastrigin


class CAVisualizer:
    """Visualizer for Cultural Algorithm optimization process."""

    def __init__(
        self,
        algorithm: CulturalAlgorithm,
        save_path: Optional[str] = None,
        fps: int = 10,
        show_population: bool = True,
        show_belief_space: bool = True,
        show_contour: bool = True,
    ) -> None:
        """Initialize the visualizer.

        Args:
            algorithm: CulturalAlgorithm instance to visualize
            save_path: Path to save video file (optional)
            fps: Frames per second for animation
            show_population: Whether to show population individuals
            show_belief_space: Whether to show belief space bounds
            show_contour: Whether to show objective function contour
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib and numpy are required for visualization. "
                "Install with: pip install matplotlib numpy"
            )

        if algorithm.dimension != 2:
            raise ValueError(
                f"Visualization currently supports 2D problems only. "
                f"Got dimension {algorithm.dimension}"
            )

        self.algorithm = algorithm
        self.save_path = save_path or "ca_optimization.mp4"
        self.fps = fps
        self.show_population = show_population
        self.show_belief_space = show_belief_space
        self.show_contour = show_contour

        # Storage for animation frames
        self.history: List[CAState] = []

    def record_state(self) -> None:
        """Record current state of algorithm for visualization."""
        if self.algorithm.best is None or self.algorithm.belief_space is None:
            return

        # Deep copy current state
        population_copy = [
            Individual(position=ind.position[:], fitness=ind.fitness)
            for ind in self.algorithm.population
        ]

        normative = self.algorithm.belief_space.normative
        situational = self.algorithm.belief_space.situational

        state = CAState(
            iteration=len(self.history),
            population=population_copy,
            normative_bounds=(
                normative.lower_bounds[:],
                normative.upper_bounds[:],
            ),
            situational_best=[
                Individual(position=ind.position[:], fitness=ind.fitness)
                for ind in situational.best_individuals
            ],
            best_individual=Individual(
                position=self.algorithm.best.position[:],
                fitness=self.algorithm.best.fitness,
            ),
        )
        self.history.append(state)

    def create_animation(self, iterations: int) -> None:
        """Run algorithm and create animation.

        Args:
            iterations: Number of iterations to run
        """
        self.history = []
        self.algorithm.initialise()
        self.record_state()

        for iteration in range(iterations):
            # Update stateful objectives
            if hasattr(self.algorithm.objective, 'update'):
                self.algorithm.objective.update(iteration)
            self.algorithm.step()
            self.record_state()

        self._render_animation()

    def _render_animation(self) -> None:
        """Render the animation from recorded history."""
        if not self.history:
            raise ValueError("No history recorded. Run create_animation() first.")

        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel("X₁", fontsize=12)
        ax.set_ylabel("X₂", fontsize=12)
        ax.set_title("Cultural Algorithm Optimization", fontsize=14, fontweight='bold')

        # Get bounds
        x_bounds = self.algorithm.bounds[0]
        y_bounds = self.algorithm.bounds[1]

        # Create contour plot of objective function
        if self.show_contour:
            x_range = np.linspace(x_bounds[0], x_bounds[1], 100)
            y_range = np.linspace(y_bounds[0], y_bounds[1], 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.algorithm.objective([X[i, j], Y[i, j]])

            contour = ax.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
            ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)
            plt.colorbar(contour, ax=ax, label='Objective Value')

        # Set axis limits
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Animation elements
        population_scatter = None
        best_scatter = None
        normative_rect = None
        situational_scatter = None
        text_info = None

        def animate(frame: int) -> None:
            nonlocal population_scatter, best_scatter, normative_rect, situational_scatter, text_info

            # Clear previous annotations
            if text_info is not None:
                text_info.remove()

            state = self.history[frame]

            # Clear previous elements
            if population_scatter is not None:
                population_scatter.remove()
            if best_scatter is not None:
                best_scatter.remove()
            if normative_rect is not None:
                normative_rect.remove()
            if situational_scatter is not None:
                situational_scatter.remove()

            # Plot population
            if self.show_population:
                pop_x = [ind.position[0] for ind in state.population]
                pop_y = [ind.position[1] for ind in state.population]
                population_scatter = ax.scatter(
                    pop_x, pop_y,
                    c='blue', alpha=0.6, s=30,
                    edgecolors='darkblue', linewidths=0.5,
                    label='Population',
                    zorder=3
                )

            # Plot best individual
            best_x = state.best_individual.position[0]
            best_y = state.best_individual.position[1]
            best_scatter = ax.scatter(
                [best_x], [best_y],
                c='red', s=200, marker='*',
                edgecolors='darkred', linewidths=2,
                label='Best Found',
                zorder=5
            )

            # Plot normative bounds
            if self.show_belief_space:
                lo_x, hi_x = state.normative_bounds[0]
                lo_y, hi_y = state.normative_bounds[1]
                width = hi_x - lo_x
                height = hi_y - lo_y

                normative_rect = patches.Rectangle(
                    (lo_x, lo_y), width, height,
                    linewidth=2, edgecolor='green',
                    facecolor='none', linestyle='--',
                    alpha=0.7, label='Normative Bounds',
                    zorder=4
                )
                ax.add_patch(normative_rect)

                # Plot situational knowledge (best examples)
                if state.situational_best:
                    sit_x = [ind.position[0] for ind in state.situational_best]
                    sit_y = [ind.position[1] for ind in state.situational_best]
                    situational_scatter = ax.scatter(
                        sit_x, sit_y,
                        c='orange', s=100, marker='s',
                        edgecolors='darkorange', linewidths=1.5,
                        label='Situational Knowledge',
                        zorder=4
                    )

            # Add iteration info
            text_info = ax.text(
                0.02, 0.98,
                f'Iteration: {state.iteration}\n'
                f'Best Fitness: {state.best_individual.fitness:.4f}\n'
                f'Population Size: {len(state.population)}\n'
                f'Normative Bounds: [{state.normative_bounds[0][0]:.2f}, {state.normative_bounds[0][1]:.2f}] × [{state.normative_bounds[1][0]:.2f}, {state.normative_bounds[1][1]:.2f}]',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                zorder=6
            )

            # Add legend
            if frame == 0:
                ax.legend(loc='upper right', fontsize=9)

        # Create animation
        anim = FuncAnimation(
            fig, animate,
            frames=len(self.history),
            interval=1000 / self.fps,
            repeat=True,
            blit=False
        )

        # Save animation
        print(f"Saving animation to {self.save_path}...")
        
        # Try to save as MP4 with ffmpeg
        if self.save_path.endswith('.mp4'):
            try:
                anim.save(
                    self.save_path,
                    writer='ffmpeg',
                    fps=self.fps,
                    bitrate=1800,
                    extra_args=['-vcodec', 'libx264']
                )
                print(f"Animation saved successfully as MP4!")
                plt.close(fig)
                return
            except Exception as e:
                print(f"Error saving with ffmpeg: {e}")
                print("Falling back to GIF format...")
                gif_path = self.save_path.replace('.mp4', '.gif')
        else:
            gif_path = self.save_path if self.save_path.endswith('.gif') else self.save_path + '.gif'
        
        # Fallback: save as GIF (requires pillow)
        try:
            anim.save(gif_path, writer='pillow', fps=self.fps)
            print(f"Animation saved as GIF: {gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Note: For video output, install ffmpeg. For GIF, ensure pillow is installed.")
            raise

        plt.close(fig)


class CAState:
    """Records state of Cultural Algorithm at a point in time."""

    def __init__(
        self,
        iteration: int,
        population: List[Individual],
        normative_bounds: Tuple[List[float], List[float]],
        situational_best: List[Individual],
        best_individual: Individual,
    ) -> None:
        """Initialize state record."""
        self.iteration = iteration
        self.population = population
        self.normative_bounds = normative_bounds  # (lower, upper) for each dim
        self.situational_best = situational_best
        self.best_individual = best_individual


def visualize_ca(
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-5.12, 5.12), (-5.12, 5.12)),
    population_size: int = 30,
    iterations: int = 50,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    **ca_kwargs
) -> None:
    """Convenience function to create CA visualization.

    Args:
        bounds: Search space bounds (must be 2D)
        population_size: Population size
        iterations: Number of iterations
        save_path: Path to save video (default: ca_optimization.mp4)
        seed: Random seed
        **ca_kwargs: Additional arguments for CulturalAlgorithm
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib and numpy are required. Install with: pip install matplotlib numpy"
        )

    # Create algorithm
    ca = CulturalAlgorithm(
        objective=rastrigin,
        bounds=bounds,
        population_size=population_size,
        seed=seed,
        **ca_kwargs
    )

    # Create visualizer
    visualizer = CAVisualizer(ca, save_path=save_path)
    visualizer.create_animation(iterations)


def main() -> None:
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Cultural Algorithm")
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of iterations"
    )
    parser.add_argument(
        "--population-size", type=int, default=30,
        help="Population size"
    )
    parser.add_argument(
        "--output", type=str, default="ca_optimization.mp4",
        help="Output video file path"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    visualize_ca(
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        population_size=args.population_size,
        iterations=args.iterations,
        save_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

