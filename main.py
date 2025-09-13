"""
Main entry point for Genetic Algorithm Optimization experiments.

This script serves as the central hub for running all optimization experiments.
Uncomment the desired experiment type to run it.

Available experiments:
- Sequential GA/PSO experiments
- Parallel GA/PSO experiments  
- Quick demos
- Comprehensive statistical analysis
- Algorithm comparisons
"""

import sys
import os

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm_optimization.experiments.sequential_experiments import (
    run_comprehensive_experiments,
    run_quick_demo
)
from genetic_algorithm_optimization.experiments.parallel_experiments import (
    run_comprehensive_parallel_experiments,
    run_parallel_vs_sequential_comparison,
    run_quick_parallel_demo
)
# Note: These experiments have been cleaned up and are no longer available
# Use the experiments in the genetic_algorithm_optimization.experiments module instead


def main():
    """
    Main function to run optimization experiments.
    
    Uncomment the experiment you want to run and comment out the others.
    """
    
    print("=" * 80)
    print("🧬 GENETIC ALGORITHM OPTIMIZATION EXPERIMENTS")
    print("=" * 80)
    print()
    
    # =============================================================================
    # QUICK DEMOS (Fast testing - recommended for first run)
    # =============================================================================
    
    # Quick sequential demo (GA + PSO on simple functions)
    # print("🚀 Running Quick Sequential Demo...")
    # run_quick_demo()
    
    # Quick parallel demo (Parallel GA on simple functions)
    # print("🚀 Running Quick Parallel Demo...")
    # run_quick_parallel_demo()
    
    # =============================================================================
    # COMPREHENSIVE EXPERIMENTS (Full statistical analysis)
    # =============================================================================
    
    # FINAL COMPREHENSIVE EXPERIMENT (ALL functions, ALL algorithms, complete statistics)
    # print("🎯 Running Final Comprehensive Experiment...")
    # run_final_comprehensive_experiment()
    
    # =============================================================================
    # STANDARD EXPERIMENT (Clean, simple, single GA + PSO per function)
    # =============================================================================
    
    # Final standard experiment (Complete GA vs PSO on all 67 functions)
    print("🎯 Running Final Standard Experiment...")
    from final_standard_experiment import run_final_standard_experiment
    run_final_standard_experiment()
    
    # Simple working example (for quick testing)
    # print("🎯 Running Simple Example...")
    # from simple_example import main as run_simple_example
    # run_simple_example()
    
    # Comprehensive sequential experiments (All GA configurations on all functions)
    # print("📊 Running Comprehensive Sequential Experiments...")
    # run_comprehensive_experiments()
    
    # Comprehensive parallel experiments (Parallel GA with detailed statistics)
    # print("📊 Running Comprehensive Parallel Experiments...")
    # run_comprehensive_parallel_experiments()
    
    # =============================================================================
    # COMPARISON EXPERIMENTS (Performance analysis)
    # =============================================================================
    
    # Parallel vs Sequential comparison (Performance benchmarking)
    # 
    
    # =============================================================================
    # CUSTOM EXPERIMENTS (Modify these for specific tests)
    # =============================================================================
    
    # Example: Run specific algorithm on specific function
    # from genetic_algorithm_optimization.sequential.algorithms import ModularGeneticAlgorithm
    # from genetic_algorithm_optimization.benchmarks.functions import get_function_by_name
    # 
    # # Get a specific function
    # sphere_func = get_function_by_name("Sphere")
    # 
    # # Create GA instance
    # ga = ModularGeneticAlgorithm(
    #     objective_function=sphere_func,
    #     bounds=sphere_func.get_bounds_for_dimension(30),
    #     dimension=30,
    #     population_size=50,
    #     crossover_prob=0.75,
    #     mutation_prob=0.01,
    #     max_evaluations=40000
    # )
    # 
    # # Run optimization
    # best_solution, best_fitness = ga.run()
    # print(f"Best solution: {best_solution}")
    # print(f"Best fitness: {best_fitness}")
    
    print()
    print("=" * 80)
    print("✅ Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
