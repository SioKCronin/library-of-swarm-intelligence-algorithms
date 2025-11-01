# Library of Nature-Inspired Optimization [WIP]

Links to original papers introducing (or meta-analysis overviews of) the following algorithms/heuristics/methods:

* Genetic Algorithms (GA)
* Particle Swarm Optimization (PSO)
* Artificial immune systems (AIS) 
* Boids
* Memetic Algorithm (MA)
* Ant Colony Optimization (ACO)
* Cultural Algorithms (CA)
* Particle Swarm Optimization (PSO)
* Self-propelled Particles
* Differential Evolution (DE)
* Bacterial Foraging Optimization
* Marriage in Honey Bees (MHB) 
* Artificial Fish School
* Bacteria Chemotaxis (BC)
* Social Cognitive Optimization (SCO)
* Artificial Bee Colony
* Glowworm Swarm Optimization (GSO)
* Honey-Bees Mating Optimization (HBMO)
* Invasive Weed Optimization (IWO)
* Shuffled Frog Leaping Algorithm (SFLA)
* Intelligent Water Drops (IWD)
* River Formation Dynamics
* Biogeography-based Optimization (BBO)
* Roach Infestation Optimization (RIO)
* Bacterial Evolutionary Algorithm (BEA)
* Cuckoo Search (CS)
* Firefly Algorithm (FA) 
* Gravitational Search Algorithm (GSA)
* [Bat Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S1877750322002903)
* Eagle Strategy
* Fireworks algorithm
* Altruism Algorithm
* Spiral Dynamic Algorithm (SDA)
* Strawberry Algorithm
* Artificial Algae Algorithm (AAA) 
* Bacterial Colony Optimization
* Flower pollination algorithm (FPA)
* [Krill Herd](https://www.sciencedirect.com/science/article/pii/S1007570412002171)
* Water Cycle Algorithm 
* Black Holes Algorithm
* Cuttlefish Algorithm
* Gases Brownian Motion Optimization
* Mine blast algorithm
* Plant Propagation Algorithm
* Social Spider Optimization (SSO)
* Spider Monkey Optimization (SMO) 
* Animal Migration Optimization (AMO) 
* Artificial Ecosystem Algorithm (AEA)
* Bird Mating Optimizer
* Forest Optimization Algorithm
* Grey Wolf Optimizer
* Lion Optimization Algorithm (LOA)
* Optics Inspired Optimization (OIO)
* The Raven Roosting Optimisation Algorithm
* Water Wave Optimization
* Collective animal behavior (CAB)
* Aritificial Chemical Process Algorithm
* Bull optimization algorithm
* Elephent herding optimization (EHO)

# Publications

* [Algorithms](http://www.mdpi.com/journal/algorithms)
* [Journal of Algorithms](https://www.sciencedirect.com/journal/journal-of-algorithms)
* [Swarm and Evolutionary Computation](https://www.journals.elsevier.com/swarm-and-evolutionary-computation/)
* [International Journal of Swarm Intelligence and Evolutionary Computation](https://www.omicsonline.org/swarm-intelligence-evolutionary-computation.php#)
* [Swarm Intelligence](https://link.springer.com/journal/11721)
* [Evolutionary Intelligence](http://www.springer.com/engineering/computational+intelligence+and+complexity/journal/12065)

# Conferences

* [GECCO](http://gecco-2018.sigevo.org/index.html/tiki-index.php?page=HomePage)

# Research teams

* [Tübingen](http://www.ra.cs.uni-tuebingen.de/links/genetisch/welcome_e.html)


## Getting Started

Install the package locally in editable mode::

```bash
pip install -e .
```

Once installed you can import `nio` from anywhere on your system.

### Using the Bat Algorithm

```python
from nio import BatAlgorithm

optimizer = BatAlgorithm(bounds=[(-5.12, 5.12)] * 5, population_size=40, seed=42)
best_position, best_value = optimizer.run(iterations=200)
print(best_value)
```

### Command-line demo

```bash
python -m nio --iterations 200 --dimension 5
```

This runs the reference implementation from Yang (2010) on a Rastrigin benchmark.
