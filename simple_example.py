#!/usr/bin/env python3
"""
Simple Example - Genetic Algorithm Optimization
==============================================

A clean, simple example showing how to use the GA and PSO algorithms.
"""

import numpy as np
from genetic_algorithm_optimization.sequential.algorithms import ModularGeneticAlgorithm, ParticleSwarmOptimization

def sphere_function(x):
    """Sphere function - global minimum at origin."""
    return np.sum(x**2)

def ackley_function(x):
    """Ackley function - multimodal with many local minima."""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def main():
    """Run a simple comparison between GA and PSO."""
    
    print("=" * 60)
    print("🧬 SIMPLE GA vs PSO COMPARISON")
    print("=" * 60)
    
    # Problem setup
    dimension = 10
    bounds = [(-5.12, 5.12) for _ in range(dimension)]
    
    # Test functions
    functions = [
        ("Sphere", sphere_function),
        ("Ackley", ackley_function)
    ]
    
    for func_name, func in functions:
        print(f"\n🔍 Testing {func_name} Function")
        print("-" * 40)
        
        # GA Configuration
        print("🧬 Running Genetic Algorithm...")
        ga = ModularGeneticAlgorithm(
            objective_function=func,
            bounds=bounds,
            dimension=dimension,
            population_size=30,
            crossover_type='arithmetic',
            mutation_type='gaussian',
            selection_type='tournament',
            replacement_type='generational',
            crossover_prob=0.75,
            mutation_prob=0.01,
            max_evaluations=10000
        )
        
        ga_solution, ga_fitness = ga.run()
        print(f"GA Best Fitness: {ga_fitness:.6f}")
        print(f"GA Best Solution: {ga_solution.genes[:5]}...")  # Show first 5 dimensions
        
        # PSO Configuration
        print("\n🐝 Running Particle Swarm Optimization...")
        pso = ParticleSwarmOptimization(
            objective_function=func,
            bounds=bounds,
            dimension=dimension,
            max_evaluations=10000,
            swarm_size=30,
            w=0.74,
            c1=1.42,
            c2=1.42
        )
        
        pso_solution, pso_fitness = pso.run()
        print(f"PSO Best Fitness: {pso_fitness:.6f}")
        print(f"PSO Best Solution: {pso_solution[:5]}...")  # Show first 5 dimensions
        
        # Comparison
        print(f"\n🏆 Comparison for {func_name}:")
        print(f"GA:  {ga_fitness:.6f}")
        print(f"PSO: {pso_fitness:.6f}")
        print(f"Winner: {'GA' if ga_fitness < pso_fitness else 'PSO'}")
        print("=" * 60)

if __name__ == "__main__":
    main()
