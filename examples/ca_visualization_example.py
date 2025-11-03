#!/usr/bin/env python3
"""
Example: Cultural Algorithm Visualization

This example demonstrates how to create animated visualizations
of the Cultural Algorithm optimization process.

Requirements:
    pip install matplotlib numpy
    (Optional: ffmpeg for MP4 video output)
"""

from nio.visualize_ca import visualize_ca, CAVisualizer
from nio import CulturalAlgorithm, rastrigin


def example_simple():
    """Simple example using the convenience function."""
    print("Creating simple CA visualization...")
    
    visualize_ca(
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        population_size=30,
        iterations=50,
        save_path="ca_simple.mp4",
        seed=42,
        accept_rate=0.2,
        influence_rate=0.3,
    )
    print("✓ Visualization saved to ca_simple.mp4")


def example_custom():
    """Example with custom algorithm configuration."""
    print("Creating custom CA visualization...")
    
    # Create algorithm with custom settings
    ca = CulturalAlgorithm(
        objective=rastrigin,
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        population_size=40,
        accept_rate=0.25,
        influence_rate=0.4,
        mutation_rate=0.15,
        num_situational=7,
        seed=42,
    )
    
    # Create visualizer with custom settings
    visualizer = CAVisualizer(
        ca,
        save_path="ca_custom.gif",
        fps=8,
        show_population=True,
        show_belief_space=True,
        show_contour=True,
    )
    
    # Run and visualize
    visualizer.create_animation(iterations=60)
    print("✓ Visualization saved to ca_custom.gif")


def example_with_custom_objective():
    """Example with a custom 2D objective function."""
    import math
    
    def sphere_2d(position):
        """2D sphere function (minimum at origin)."""
        return sum(x * x for x in position)
    
    def rosenbrock_2d(position):
        """2D Rosenbrock function."""
        x, y = position[0], position[1]
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    
    print("Creating CA visualization with custom objective (Rosenbrock)...")
    
    ca = CulturalAlgorithm(
        objective=rosenbrock_2d,
        bounds=((-2.0, 2.0), (-1.0, 3.0)),
        population_size=35,
        seed=42,
    )
    
    visualizer = CAVisualizer(
        ca,
        save_path="ca_rosenbrock.mp4",
        fps=10,
    )
    
    visualizer.create_animation(iterations=80)
    print("✓ Visualization saved to ca_rosenbrock.mp4")


if __name__ == "__main__":
    import sys
    
    print("Cultural Algorithm Visualization Examples")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "simple":
            example_simple()
        elif example_name == "custom":
            example_custom()
        elif example_name == "rosenbrock":
            example_with_custom_objective()
        else:
            print(f"Unknown example: {example_name}")
            print("Available: simple, custom, rosenbrock")
    else:
        print("\nRunning simple example...")
        print("(Use 'simple', 'custom', or 'rosenbrock' as argument for different examples)")
        print()
        example_simple()

