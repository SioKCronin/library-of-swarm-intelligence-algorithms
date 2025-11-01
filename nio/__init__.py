"""nio: Nature-inspired optimization toolkit.

Currently includes:
- BatAlgorithm (Yang, 2010)

Usage example::

    from nio import BatAlgorithm

    optimizer = BatAlgorithm()
    solution, value = optimizer.run(200)

"""

from .bat import BatAlgorithm, Bat, rastrigin

__all__ = ["BatAlgorithm", "Bat", "rastrigin"]
