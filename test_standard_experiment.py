#!/usr/bin/env python3
"""
Test Standard Experiment - Quick test with 2 functions
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm_optimization.sequential.algorithms import ModularGeneticAlgorithm, ParticleSwarmOptimization
from genetic_algorithm_optimization.benchmarks.manual_functions import BENCHMARK_FUNCTIONS

def test_standard_experiment():
    """Test the standard experiment with just 2 functions."""
    
    print("=" * 60)
    print("🧪 TESTING STANDARD EXPERIMENT - 2 FUNCTIONS")
    print("=" * 60)
    
    # Test parameters (smaller for testing)
    num_runs = 3
    max_evaluations = 5000
    population_size = 20
    
    # GA Configuration (as specified by teacher)
    ga_config = {
        'population_size': population_size,
        'crossover_type': 'arithmetic',
        'mutation_type': 'gaussian',
        'selection_type': 'tournament',
        'replacement_type': 'generational',
        'crossover_prob': 0.75,
        'mutation_prob': 0.01,
        'max_evaluations': max_evaluations
    }
    
    # PSO Configuration (as specified by teacher)
    pso_config = {
        'swarm_size': population_size,
        'w': 0.74,
        'c1': 1.42,
        'c2': 1.42,
        'max_evaluations': max_evaluations
    }
    
    # Test functions
    test_functions = ['sphere', 'ackley']
    
    for func_name in test_functions:
        if func_name not in BENCHMARK_FUNCTIONS:
            print(f"Function {func_name} not found!")
            continue
            
        func_info = BENCHMARK_FUNCTIONS[func_name]
        print(f"\n🔍 Testing: {func_name}")
        print(f"Category: {func_info['type']}, Dimension Type: {func_info['dim_type']}")
        
        # Determine dimension
        if func_info['dim_type'] == 'n-dim':
            dimension = 30
        elif isinstance(func_info['dim_type'], int):
            dimension = func_info['dim_type']
        else:
            dimension = 30
        
        # Create bounds list
        bounds = func_info['bounds']
        if isinstance(bounds, tuple):
            bounds_list = [bounds] * dimension
        else:
            bounds_list = bounds
        
        func = func_info['func']
        
        # Test GA
        print("🧬 Testing GA...")
        try:
            ga = ModularGeneticAlgorithm(
                objective_function=func,
                bounds=bounds_list,
                dimension=dimension,
                **ga_config
            )
            best_solution, best_fitness = ga.run()
            print(f"GA Result: {best_fitness:.6f}")
        except Exception as e:
            print(f"GA Error: {e}")
        
        # Test PSO
        print("🐝 Testing PSO...")
        try:
            pso = ParticleSwarmOptimization(
                objective_function=func,
                bounds=bounds_list,
                dimension=dimension,
                **pso_config
            )
            best_solution, best_fitness = pso.run()
            print(f"PSO Result: {best_fitness:.6f}")
        except Exception as e:
            print(f"PSO Error: {e}")
        
        print("-" * 40)
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_standard_experiment()
