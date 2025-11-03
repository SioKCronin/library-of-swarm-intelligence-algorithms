"""nio: Nature-inspired optimization toolkit.

Currently includes:
- BatAlgorithm (Yang, 2010)
- PhilippineEagleOptimization
- ContractingOptimum benchmark

Usage example::

    from nio import BatAlgorithm, PhilippineEagleOptimization, ContractingOptimum

    optimizer = BatAlgorithm()
    solution, value = optimizer.run(200)

    peo = PhilippineEagleOptimization()
    solution, value = peo.run(200)

    # Dynamic benchmark with contracting optimum
    benchmark = ContractingOptimum(bounds=[(-5.12, 5.12)] * 5, max_iterations=200)
    optimizer = BatAlgorithm(objective=benchmark)
    solution, value = optimizer.run(200)

"""

from .bat import BatAlgorithm, Bat, rastrigin
from .philippine_eagle import PhilippineEagleOptimization, Eagle, Operator, Phase
from .benchmarks import ContractingOptimum, contracting_optimum

__all__ = [
    "BatAlgorithm",
    "Bat",
    "rastrigin",
    "PhilippineEagleOptimization",
    "Eagle",
    "Operator",
    "Phase",
    "ContractingOptimum",
    "contracting_optimum",
]
