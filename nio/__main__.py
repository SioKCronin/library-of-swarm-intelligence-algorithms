"""Command line entry point for nio.

Example:
    python -m nio --iterations 200 --dimension 5
"""

from __future__ import annotations

import argparse

from .bat import BatAlgorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bat Algorithm demo")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations to run")
    parser.add_argument(
        "--dimension",
        type=int,
        default=5,
        help="Dimensionality of the search space (bounds remain [-5.12, 5.12])",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bounds = [(-5.12, 5.12)] * args.dimension
    optimizer = BatAlgorithm(bounds=bounds, population_size=40, seed=args.seed)
    position, value = optimizer.run(iterations=args.iterations)
    print("Best value:", value)
    print("Best position:", [round(x, 4) for x in position])


if __name__ == "__main__":
    main()
