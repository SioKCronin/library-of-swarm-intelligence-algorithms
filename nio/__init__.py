"""nio: Nature-inspired optimization toolkit.

Currently includes:
- BatAlgorithm (Yang, 2010)
- PhilippineEagleOptimization
- CulturalAlgorithm (Reynolds, 1994)
- ContractingOptimum benchmark

Usage example::

    from nio import BatAlgorithm, PhilippineEagleOptimization, CulturalAlgorithm
    from nio import ContractingOptimum

    optimizer = BatAlgorithm()
    solution, value = optimizer.run(200)

    peo = PhilippineEagleOptimization()
    solution, value = peo.run(200)

    ca = CulturalAlgorithm()
    solution, value = ca.run(200)

    # Dynamic benchmark with contracting optimum
    benchmark = ContractingOptimum(bounds=[(-5.12, 5.12)] * 5, max_iterations=200)
    optimizer = BatAlgorithm(objective=benchmark)
    solution, value = optimizer.run(200)

"""

from .bat import BatAlgorithm, Bat, rastrigin
from .philippine_eagle import PhilippineEagleOptimization, Eagle, Operator, Phase
from .cultural import CulturalAlgorithm, Individual, BeliefSpace, NormativeKnowledge, SituationalKnowledge
from .benchmarks import ContractingOptimum, contracting_optimum

__all__ = [
    "BatAlgorithm",
    "Bat",
    "rastrigin",
    "PhilippineEagleOptimization",
    "Eagle",
    "Operator",
    "Phase",
    "CulturalAlgorithm",
    "Individual",
    "BeliefSpace",
    "NormativeKnowledge",
    "SituationalKnowledge",
    "ContractingOptimum",
    "contracting_optimum",
]
