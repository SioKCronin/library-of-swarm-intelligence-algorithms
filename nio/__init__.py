"""nio: Nature-inspired optimization toolkit.

Currently includes:
- BatAlgorithm (Yang, 2010)
- PhilippineEagleOptimization

Usage example::

    from nio import BatAlgorithm, PhilippineEagleOptimization

    optimizer = BatAlgorithm()
    solution, value = optimizer.run(200)

    peo = PhilippineEagleOptimization()
    solution, value = peo.run(200)

"""

from .bat import BatAlgorithm, Bat, rastrigin
from .philippine_eagle import PhilippineEagleOptimization, Eagle, Operator, Phase

__all__ = [
    "BatAlgorithm",
    "Bat",
    "rastrigin",
    "PhilippineEagleOptimization",
    "Eagle",
    "Operator",
    "Phase",
]
