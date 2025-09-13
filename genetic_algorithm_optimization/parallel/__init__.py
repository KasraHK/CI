"""
Parallel optimization algorithms.

This module contains parallel implementations of optimization algorithms
with parallel fitness evaluation capabilities for improved performance.
"""

from .algorithms import (
    ParallelGeneticAlgorithm,
    ParallelEvaluator
)

__all__ = [
    "ParallelGeneticAlgorithm",
    "ParallelEvaluator"
]
