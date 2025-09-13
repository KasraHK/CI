#!/usr/bin/env python3
"""
Final Standard Experiment - Complete GA vs PSO Comparison
========================================================

This script runs the complete standard experiment with all 67 benchmark functions
using the exact parameters specified by the teacher:

- GA: Mutation rate = 0.01, Crossover probability = 0.75
- PSO: w = 0.74, c1 = 1.42, c2 = 1.42  
- Population size = 50 for both algorithms
- End condition: 40,000 function evaluations
- Dimensions: 2D functions run with dimension = 2, N-dimensional functions run with dimension = 30
- Runs: 20 times per algorithm per function for statistics
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

def run_final_standard_experiment():
    """Run the complete final standard experiment with all 67 functions."""
    
    print("=" * 80)
    print("🎯 FINAL STANDARD EXPERIMENT - COMPLETE GA vs PSO COMPARISON")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Experiment parameters (as specified by teacher)
    num_runs = 20
    max_evaluations = 40000
    population_size = 50
    
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
    
    # Results storage
    results = []
    
    # Get all functions
    all_functions = list(BENCHMARK_FUNCTIONS.items())
    total_functions = len(all_functions)
    
    print(f"Testing {total_functions} functions with {num_runs} runs each")
    print(f"GA Config: {ga_config}")
    print(f"PSO Config: {pso_config}")
    print("=" * 80)
    
    for i, (func_name, func_info) in enumerate(all_functions, 1):
        print(f"\n[{i}/{total_functions}] Testing: {func_name}")
        print(f"Category: {func_info['type']}, Dimension Type: {func_info['dim_type']}")
        print(f"Bounds: {func_info['bounds']}")
        print(f"Expected Global Minimum: {func_info['global_minimum']}")
        print("-" * 60)
        
        # Determine dimension based on function type
        if func_info['dim_type'] == 'n-dim':
            dimension = 30
        elif isinstance(func_info['dim_type'], int):
            dimension = func_info['dim_type']
        else:
            dimension = 30  # Default fallback
        
        # Create bounds list
        bounds = func_info['bounds']
        if isinstance(bounds, tuple):
            bounds_list = [bounds] * dimension
        else:
            bounds_list = bounds
        
        # Get function
        func = func_info['func']
        
        # Run GA
        print("🧬 Running GA...")
        ga_fitness_values = []
        ga_times = []
        
        for run in range(num_runs):
            try:
                ga = ModularGeneticAlgorithm(
                    objective_function=func,
                    bounds=bounds_list,
                    dimension=dimension,
                    **ga_config
                )
                
                start_time = datetime.now()
                best_solution, best_fitness = ga.run()
                end_time = datetime.now()
                
                # Check for invalid fitness values
                if not np.isfinite(best_fitness):
                    print(f"  Run {run+1}/{num_runs}: WARNING - Invalid fitness: {best_fitness}")
                    best_fitness = float('inf')
                elif abs(best_fitness) > 1e10:
                    print(f"  Run {run+1}/{num_runs}: WARNING - Extremely large value: {best_fitness}")
                    best_fitness = float('inf')
                
                ga_fitness_values.append(best_fitness)
                ga_times.append((end_time - start_time).total_seconds())
                
                if run < 3:  # Show first 3 runs
                    print(f"  Run {run+1}/{num_runs}: {best_fitness:.6f}")
                
            except Exception as e:
                print(f"  Run {run+1}/{num_runs}: ERROR - {e}")
                ga_fitness_values.append(float('inf'))
                ga_times.append(0)
        
        # Run PSO
        print("🐝 Running PSO...")
        pso_fitness_values = []
        pso_times = []
        
        for run in range(num_runs):
            try:
                pso = ParticleSwarmOptimization(
                    objective_function=func,
                    bounds=bounds_list,
                    dimension=dimension,
                    **pso_config
                )
                
                start_time = datetime.now()
                best_solution, best_fitness = pso.run()
                end_time = datetime.now()
                
                # Check for invalid fitness values
                if not np.isfinite(best_fitness):
                    print(f"  Run {run+1}/{num_runs}: WARNING - Invalid fitness: {best_fitness}")
                    best_fitness = float('inf')
                elif abs(best_fitness) > 1e10:
                    print(f"  Run {run+1}/{num_runs}: WARNING - Extremely large value: {best_fitness}")
                    best_fitness = float('inf')
                
                pso_fitness_values.append(best_fitness)
                pso_times.append((end_time - start_time).total_seconds())
                
                if run < 3:  # Show first 3 runs
                    print(f"  Run {run+1}/{num_runs}: {best_fitness:.6f}")
                
            except Exception as e:
                print(f"  Run {run+1}/{num_runs}: ERROR - {e}")
                pso_fitness_values.append(float('inf'))
                pso_times.append(0)
        
        # Calculate statistics (filter out inf values)
        ga_finite = [f for f in ga_fitness_values if np.isfinite(f)]
        pso_finite = [f for f in pso_fitness_values if np.isfinite(f)]
        
        ga_stats = {
            'mean': np.mean(ga_finite) if ga_finite else float('inf'),
            'std': np.std(ga_finite) if len(ga_finite) > 1 else 0.0,
            'min': np.min(ga_finite) if ga_finite else float('inf'),
            'max': np.max(ga_finite) if ga_finite else float('inf'),
            'success_rate': len(ga_finite) / num_runs * 100
        }
        
        pso_stats = {
            'mean': np.mean(pso_finite) if pso_finite else float('inf'),
            'std': np.std(pso_finite) if len(pso_finite) > 1 else 0.0,
            'min': np.min(pso_finite) if pso_finite else float('inf'),
            'max': np.max(pso_finite) if pso_finite else float('inf'),
            'success_rate': len(pso_finite) / num_runs * 100
        }
        
        # Store results
        results.append({
            'Function': func_name,
            'Category': func_info['type'],
            'Dimension': dimension,
            'Bounds': str(bounds),
            'Expected_Min': func_info['global_minimum'],
            'GA_Mean': ga_stats['mean'],
            'GA_Std': ga_stats['std'],
            'GA_Min': ga_stats['min'],
            'GA_Max': ga_stats['max'],
            'GA_Success_Rate': ga_stats['success_rate'],
            'PSO_Mean': pso_stats['mean'],
            'PSO_Std': pso_stats['std'],
            'PSO_Min': pso_stats['min'],
            'PSO_Max': pso_stats['max'],
            'PSO_Success_Rate': pso_stats['success_rate']
        })
        
        # Print summary
        print(f"\n📊 Results for {func_name}:")
        print(f"GA:  Mean={ga_stats['mean']:.6f}, Std={ga_stats['std']:.6f}, Success={ga_stats['success_rate']:.1f}%")
        print(f"PSO: Mean={pso_stats['mean']:.6f}, Std={pso_stats['std']:.6f}, Success={pso_stats['success_rate']:.1f}%")
        print("=" * 60)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/final_standard_results_{timestamp}.csv"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    df.to_csv(filename, index=False)
    
    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print(f"📁 Results saved to: {filename}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total functions tested: {total_functions}")
    print(f"🔄 Total runs: {total_functions * 2 * num_runs}")
    
    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"GA Success Rate: {df['GA_Success_Rate'].mean():.1f}%")
    print(f"PSO Success Rate: {df['PSO_Success_Rate'].mean():.1f}%")
    
    # Count winners
    ga_wins = 0
    pso_wins = 0
    ties = 0
    
    for _, row in df.iterrows():
        if row['GA_Mean'] < row['PSO_Mean']:
            ga_wins += 1
        elif row['PSO_Mean'] < row['GA_Mean']:
            pso_wins += 1
        else:
            ties += 1
    
    print(f"🏆 GA Wins: {ga_wins}")
    print(f"🏆 PSO Wins: {pso_wins}")
    print(f"🤝 Ties: {ties}")
    
    return df

if __name__ == "__main__":
    results = run_final_standard_experiment()
