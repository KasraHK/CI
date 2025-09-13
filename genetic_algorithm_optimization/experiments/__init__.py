"""
Experiment modules for genetic algorithm optimization.

This module contains experiment runners for both sequential and parallel
optimization algorithms with comprehensive statistical analysis.
"""

from .sequential_experiments import run_comprehensive_experiments, run_quick_demo
from .parallel_experiments import run_comprehensive_parallel_experiments, run_parallel_vs_sequential_comparison, run_quick_parallel_demo

__all__ = [
    "run_comprehensive_experiments",
    "run_quick_demo",
    "run_comprehensive_parallel_experiments", 
    "run_parallel_vs_sequential_comparison",
    "run_quick_parallel_demo"
]
