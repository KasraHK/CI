"""
Genetic Algorithm Optimization Package.

A comprehensive implementation of Genetic Algorithms and Particle Swarm Optimization
with parallel processing capabilities for continuous optimization problems.

This package provides:
- Modular GA implementation with various operators
- Parallel fitness evaluation for improved performance
- Comprehensive benchmark function library
- Statistical analysis and visualization tools
- CSV result export and analysis

Author: Your Name
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core algorithm imports
from .sequential.algorithms import (
    ModularGeneticAlgorithm,
    AdaptiveGeneticAlgorithm,
    ParticleSwarmOptimization
)

from .parallel.algorithms import (
    ParallelGeneticAlgorithm,
    ParallelEvaluator
)

# Core components
from .core.chromosome import Chromosome
from .core.mutations import MutationFactory
from .core.crossovers import CrossoverFactory
from .core.selection import SelectionFactory
from .core.replacement import ReplacementFactory

# Benchmark functions
from .benchmarks import (
    BenchmarkFunction,
    get_functions_by_category,
    get_function_by_name,
    get_all_functions,
    benchmark_functions
)

# Utilities
from .utils.results_manager import ResultsManager
from .utils.visualize_results import (
    plot_convergence_curves,
    plot_algorithm_comparison,
    create_summary_dashboard
)

# Main experiment runners
from .experiments.sequential_experiments import run_comprehensive_experiments, run_quick_demo
from .experiments.parallel_experiments import run_comprehensive_parallel_experiments, run_parallel_vs_sequential_comparison

__all__ = [
    # Core algorithms
    "ModularGeneticAlgorithm",
    "AdaptiveGeneticAlgorithm", 
    "ParticleSwarmOptimization",
    "ParallelGeneticAlgorithm",
    "ParallelEvaluator",
    
    # Core components
    "Chromosome",
    "MutationFactory",
    "CrossoverFactory", 
    "SelectionFactory",
    "ReplacementFactory",
    
    # Benchmark functions
    "BenchmarkFunction",
    "get_functions_by_category",
    "get_function_by_name",
    "UNIMODAL_FUNCTIONS",
    "MULTIMODAL_FUNCTIONS",
    "ALL_FUNCTIONS",
    
    # Utilities
    "ResultsManager",
    "plot_convergence_curves",
    "plot_algorithm_comparison", 
    "create_summary_dashboard",
    
    # Experiments
    "run_sequential_experiments",
    "run_parallel_experiments",
]
