"""
Sequential optimization algorithms.

This module contains sequential implementations of optimization algorithms:
- Modular Genetic Algorithm
- Adaptive Genetic Algorithm
- Particle Swarm Optimization
"""

from .algorithms import (
    ModularGeneticAlgorithm,
    AdaptiveGeneticAlgorithm,
    ParticleSwarmOptimization
)

__all__ = [
    "ModularGeneticAlgorithm",
    "AdaptiveGeneticAlgorithm", 
    "ParticleSwarmOptimization"
]
