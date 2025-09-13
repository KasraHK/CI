"""
Parallel optimization experiment runner with enhanced statistical analysis.

This script runs comprehensive experiments using parallel fitness evaluation
and provides detailed statistical analysis including mean, median, std, min, max.
"""

import numpy as np
import pandas as pd
from scipy import stats
from genetic_algorithm_optimization.sequential.algorithms import ModularGeneticAlgorithm, AdaptiveGeneticAlgorithm, ParticleSwarmOptimization
from genetic_algorithm_optimization.parallel.algorithms import ParallelGeneticAlgorithm
from genetic_algorithm_optimization.benchmarks.manual_functions import BENCHMARK_FUNCTIONS
from genetic_algorithm_optimization.utils.results_manager import ResultsManager


def calculate_comprehensive_statistics(fitness_values):
    """
    Calculate comprehensive statistics for fitness values.
    
    Args:
        fitness_values: List of fitness values
        
    Returns:
        dict: Comprehensive statistics
    """
    if not fitness_values:
        return {}
    
    fitness_array = np.array(fitness_values)
    
    return {
        'count': len(fitness_values),
        'mean': np.mean(fitness_array),
        'median': np.median(fitness_array),
        'std': np.std(fitness_array),
        'min': np.min(fitness_array),
        'max': np.max(fitness_array),
        'q1': np.percentile(fitness_array, 25),
        'q3': np.percentile(fitness_array, 75),
        'iqr': np.percentile(fitness_array, 75) - np.percentile(fitness_array, 25),
        'skewness': stats.skew(fitness_array),
        'kurtosis': stats.kurtosis(fitness_array),
        'coefficient_of_variation': np.std(fitness_array) / np.mean(fitness_array) if np.mean(fitness_array) != 0 else 0
    }


def run_parallel_algorithm_experiment(algorithm_class, algorithm_name, benchmark_func, 
                                    dimension, num_runs=20, **algorithm_params):
    """
    Run a parallel algorithm experiment with comprehensive statistics.
    
    Args:
        algorithm_class: The algorithm class to use
        algorithm_name: Name of the algorithm for display
        benchmark_func: The benchmark function object
        dimension: Problem dimension
        num_runs: Number of independent runs
        **algorithm_params: Additional parameters for the algorithm
    
    Returns:
        dict: Comprehensive results including detailed statistics
    """
    print(f"\n{'='*80}")
    print(f"Running {algorithm_name} on {benchmark_func.name} function")
    print(f"Category: {benchmark_func.category}, Dimension: {dimension}, Runs: {num_runs}")
    print(f"Parameters: {algorithm_params}")
    print(f"{'='*80}")
    
    # Get bounds for the specified dimension
    bounds = benchmark_func.get_bounds_for_dimension(dimension)
    
    fitness_values = []
    run_results = []
    execution_times = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Initialize algorithm
        algorithm = algorithm_class(
            objective_function=benchmark_func.function,
            bounds=bounds,
            dimension=dimension,
            **algorithm_params
        )
        
        # Time the execution
        import time
        start_time = time.time()
        
        # Run algorithm
        best_solution, best_fitness = algorithm.run()
        
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        
        # Collect statistics
        stats = algorithm.get_statistics()
        
        run_result = {
            'run': run + 1,
            'best_fitness': best_fitness,
            'mean_fitness': stats['mean_fitness'],
            'std_fitness': stats['std_fitness'],
            'worst_fitness': stats['worst_fitness'],
            'evaluations': stats['evaluations'],
            'execution_time': execution_time,
            'best_solution': best_solution,
            'parameters': algorithm_params
        }
        
        run_results.append(run_result)
        fitness_values.append(best_fitness)
        
        print(f"Best fitness: {best_fitness:.6f}, Time: {execution_time:.2f}s")
    
    # Calculate comprehensive statistics
    fitness_stats = calculate_comprehensive_statistics(fitness_values)
    time_stats = calculate_comprehensive_statistics(execution_times)
    
    # Calculate success rate (within 1% of global minimum)
    global_min = benchmark_func.global_minimum
    tolerance = abs(global_min) * 0.01 if global_min != 0 else 0.01
    success_count = sum(1 for f in fitness_values if f <= global_min + tolerance)
    success_rate = success_count / num_runs
    
    print(f"\n{'-'*80}")
    print(f"COMPREHENSIVE RESULTS for {algorithm_name} on {benchmark_func.name}:")
    print(f"{'-'*80}")
    print(f"FITNESS STATISTICS:")
    print(f"  Count: {fitness_stats['count']}")
    print(f"  Mean:   {fitness_stats['mean']:.6f}")
    print(f"  Median: {fitness_stats['median']:.6f}")
    print(f"  Std:    {fitness_stats['std']:.6f}")
    print(f"  Min:    {fitness_stats['min']:.6f}")
    print(f"  Max:    {fitness_stats['max']:.6f}")
    print(f"  Q1:     {fitness_stats['q1']:.6f}")
    print(f"  Q3:     {fitness_stats['q3']:.6f}")
    print(f"  IQR:    {fitness_stats['iqr']:.6f}")
    print(f"  Skewness: {fitness_stats['skewness']:.4f}")
    print(f"  Kurtosis: {fitness_stats['kurtosis']:.4f}")
    print(f"  CoV:    {fitness_stats['coefficient_of_variation']:.4f}")
    print(f"\nEXECUTION TIME STATISTICS:")
    print(f"  Mean:   {time_stats['mean']:.2f}s")
    print(f"  Median: {time_stats['median']:.2f}s")
    print(f"  Std:    {time_stats['std']:.2f}s")
    print(f"  Min:    {time_stats['min']:.2f}s")
    print(f"  Max:    {time_stats['max']:.2f}s")
    print(f"\nSUCCESS RATE: {success_rate:.2%}")
    print(f"{'-'*80}")
    
    return {
        'algorithm_name': algorithm_name,
        'function_name': benchmark_func.name,
        'function_category': benchmark_func.category,
        'dimension': dimension,
        'num_runs': num_runs,
        'fitness_statistics': fitness_stats,
        'time_statistics': time_stats,
        'success_rate': success_rate,
        'run_results': run_results,
        'parameters': algorithm_params
    }


def run_parallel_vs_sequential_comparison():
    """Run comparison between parallel and sequential evaluation."""
    print("\n" + "="*80)
    print("PARALLEL vs SEQUENTIAL EVALUATION COMPARISON")
    print("="*80)
    
    # Test functions
    test_functions = []
    
    # Create function wrappers for test functions
    class FunctionWrapper:
        def __init__(self, name, func, bounds, category):
            self.name = name
            self.function = func
            self.bounds = bounds
            self.category = category
            self.description = f"{name} function"
            self.global_minimum = 0.0  # Most functions have 0 as global minimum
        
        def get_bounds_for_dimension(self, dim):
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                return [self.bounds] * dim
            return self.bounds
    
    # Get specific functions for testing
    test_function_names = ["sphere", "ackley", "rastrigin"]
    for name in test_function_names:
        if name in BENCHMARK_FUNCTIONS:
            info = BENCHMARK_FUNCTIONS[name]
            bounds = info["bounds"]
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # For 30D experiments, we need to repeat the bounds for each dimension
                bounds_list = [bounds] * 30  # Use 30 as default dimension
            else:
                bounds_list = bounds
                
            func_wrapper = FunctionWrapper(name, info["func"], bounds_list, info["type"])
            test_functions.append(func_wrapper)
    
    dimension = 20
    population_size = 100
    
    for benchmark_func in test_functions:
        print(f"\nTesting {benchmark_func.name} function...")
        
        # Create test population
        bounds = benchmark_func.get_bounds_for_dimension(dimension)
        from chromosome import Chromosome
        population = [Chromosome.random(bounds, dimension) for _ in range(population_size)]
        
        # Benchmark parallel vs sequential
        results = benchmark_parallel_vs_sequential(benchmark_func.function, population)
        
        print(f"Speedup: {results['speedup']:.2f}x")
        print(f"Efficiency: {results['efficiency']:.2%}")


def run_comprehensive_parallel_experiments():
    """Run comprehensive experiments with parallel processing."""
    
    # Initialize results manager
    results_manager = ResultsManager("results_parallel")
    
    # Experiment configuration
    dimension = 30
    num_runs = 20
    
    # Algorithm configurations (including parallel versions)
    algorithm_configurations = [
        # Sequential algorithms
        {
            'name': 'GA-Rank-Arithmetic',
            'class': ModularGeneticAlgorithm,
            'params': {
                'crossover_type': 'arithmetic',
                'mutation_type': 'gaussian',
                'selection_type': 'rank',
                'replacement_type': 'generational',
                'population_size': 50,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01
            }
        },
        {
            'name': 'PSO',
            'class': ParticleSwarmOptimization,
            'params': {
                'swarm_size': 50,
                'w': 0.74,
                'c1': 1.42,
                'c2': 1.42
            }
        },
        # Parallel algorithms
        {
            'name': 'Parallel-GA-Rank-Arithmetic',
            'class': ParallelGeneticAlgorithm,
            'params': {
                'crossover_type': 'arithmetic',
                'mutation_type': 'gaussian',
                'selection_type': 'rank',
                'replacement_type': 'generational',
                'population_size': 50,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01,
                'num_processes': 4
            }
        },
        {
            'name': 'Parallel-GA-Tournament-WholeArithmetic',
            'class': ParallelGeneticAlgorithm,
            'params': {
                'crossover_type': 'whole_arithmetic',
                'mutation_type': 'gaussian',
                'selection_type': 'tournament',
                'replacement_type': 'generational',
                'population_size': 50,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01,
                'num_processes': 4
            }
        }
    ]
    
    # Get functions to test
    test_functions = get_nd_functions()[:5]  # Test first 5 functions
    
    # Store all results
    all_results = {}
    
    print("COMPREHENSIVE PARALLEL OPTIMIZATION EXPERIMENTS")
    print("="*80)
    print(f"Testing {len(test_functions)} functions with {len(algorithm_configurations)} algorithms")
    print(f"Dimension: {dimension}, Runs per experiment: {num_runs}")
    print("="*80)
    
    # Run experiments
    for benchmark_func in test_functions:
        print(f"\n\nTESTING FUNCTION: {benchmark_func.name} ({benchmark_func.category})")
        print(f"Global minimum: {benchmark_func.global_minimum}")
        
        function_results = {}
        
        for config in algorithm_configurations:
            try:
                result = run_parallel_algorithm_experiment(
                    config['class'],
                    config['name'],
                    benchmark_func,
                    dimension,
                    num_runs,
                    **config['params']
                )
                
                # Save individual experiment results
                results_manager.save_experiment_results(
                    "parallel_study",
                    config['name'],
                    benchmark_func.name,
                    result['run_results'],
                    {
                        'function_category': benchmark_func.category,
                        'dimension': dimension,
                        'global_minimum': benchmark_func.global_minimum,
                        'algorithm_parameters': config['params'],
                        'fitness_statistics': result['fitness_statistics'],
                        'time_statistics': result['time_statistics'],
                        'success_rate': result['success_rate']
                    }
                )
                
                function_results[config['name']] = {
                    'fitness_statistics': result['fitness_statistics'],
                    'time_statistics': result['time_statistics'],
                    'success_rate': result['success_rate'],
                    'parameters': config['params']
                }
                
            except Exception as e:
                print(f"Error running {config['name']} on {benchmark_func.name}: {str(e)}")
                continue
        
        all_results[benchmark_func.name] = function_results
    
    # Save comparison results
    results_manager.save_comparison_results("parallel_study", all_results)
    
    # Generate summary report
    results_manager.generate_summary_report("parallel_study")
    
    return all_results


def run_quick_parallel_demo():
    """Run a quick demo with parallel processing."""
    
    results_manager = ResultsManager("results_parallel")
    
    # Quick demo configuration
    dimension = 15
    num_runs = 3
    
    # Test key functions
    demo_functions = []
    
    # Create function wrappers for demo functions
    class FunctionWrapper:
        def __init__(self, name, func, bounds, category):
            self.name = name
            self.function = func
            self.bounds = bounds
            self.category = category
            self.description = f"{name} function"
            self.global_minimum = 0.0  # Most functions have 0 as global minimum
        
        def get_bounds_for_dimension(self, dim):
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                return [self.bounds] * dim
            return self.bounds
    
    # Get specific functions for demo
    demo_function_names = ["sphere", "ackley", "rastrigin"]
    for name in demo_function_names:
        if name in BENCHMARK_FUNCTIONS:
            info = BENCHMARK_FUNCTIONS[name]
            bounds = info["bounds"]
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # For 30D experiments, we need to repeat the bounds for each dimension
                bounds_list = [bounds] * 30  # Use 30 as default dimension
            else:
                bounds_list = bounds
                
            func_wrapper = FunctionWrapper(name, info["func"], bounds_list, info["type"])
            demo_functions.append(func_wrapper)
    
    # Test parallel algorithms
    demo_configurations = [
        {
            'name': 'GA-Rank-Arithmetic',
            'class': ModularGeneticAlgorithm,
            'params': {
                'crossover_type': 'arithmetic',
                'mutation_type': 'gaussian',
                'selection_type': 'rank',
                'replacement_type': 'generational',
                'population_size': 30,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01
            }
        },
        {
            'name': 'Parallel-GA-Rank-Arithmetic',
            'class': ParallelGeneticAlgorithm,
            'params': {
                'crossover_type': 'arithmetic',
                'mutation_type': 'gaussian',
                'selection_type': 'rank',
                'replacement_type': 'generational',
                'population_size': 30,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01,
                'num_processes': 4
            }
        }
    ]
    
    print("QUICK PARALLEL DEMO")
    print("="*60)
    print(f"Testing {len(demo_functions)} functions with {len(demo_configurations)} algorithms")
    print(f"Dimension: {dimension}, Runs per experiment: {num_runs}")
    print("="*60)
    
    all_results = {}
    
    for benchmark_func in demo_functions:
        print(f"\n\nTESTING FUNCTION: {benchmark_func.name}")
        
        function_results = {}
        
        for config in demo_configurations:
            try:
                result = run_parallel_algorithm_experiment(
                    config['class'],
                    config['name'],
                    benchmark_func,
                    dimension,
                    num_runs,
                    **config['params']
                )
                
                function_results[config['name']] = {
                    'fitness_statistics': result['fitness_statistics'],
                    'time_statistics': result['time_statistics'],
                    'success_rate': result['success_rate']
                }
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
        all_results[benchmark_func.name] = function_results
    
    # Save results
    results_manager.save_comparison_results("quick_parallel_demo", all_results)
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("Running quick parallel demo...")
        run_quick_parallel_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("Running parallel vs sequential benchmark...")
        run_parallel_vs_sequential_comparison()
    else:
        print("Running comprehensive parallel experiments...")
        run_comprehensive_parallel_experiments()
