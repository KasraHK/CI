"""
Comprehensive optimization experiment runner with CSV output.

This script runs extensive experiments on unimodal and multimodal benchmark functions
using different GA configurations and saves all results to CSV files.
"""

import numpy as np
from genetic_algorithm_optimization.sequential.algorithms import ModularGeneticAlgorithm, AdaptiveGeneticAlgorithm, ParticleSwarmOptimization
from genetic_algorithm_optimization.benchmarks.manual_functions import BENCHMARK_FUNCTIONS
from genetic_algorithm_optimization.utils.results_manager import ResultsManager


def run_algorithm_experiment(algorithm_class, algorithm_name, benchmark_func, 
                           dimension, num_runs=20, **algorithm_params):
    """
    Run an algorithm multiple times and collect comprehensive statistics.
    
    Args:
        algorithm_class: The algorithm class to use
        algorithm_name: Name of the algorithm for display
        benchmark_func: The benchmark function object
        dimension: Problem dimension
        num_runs: Number of independent runs
        **algorithm_params: Additional parameters for the algorithm
    
    Returns:
        dict: Comprehensive results including statistics and metadata
    """
    print(f"\n{'='*80}")
    print(f"Running {algorithm_name} on {benchmark_func.name} function")
    print(f"Category: {benchmark_func.category}, Dimension: {dimension}, Runs: {num_runs}")
    print(f"Parameters: {algorithm_params}")
    print(f"{'='*80}")
    
    # Get bounds for the specified dimension
    bounds = benchmark_func.get_bounds_for_dimension(dimension)
    
    fitness_values = []
    convergence_data = []
    run_results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Initialize algorithm
        algorithm = algorithm_class(
            objective_function=benchmark_func.function,
            bounds=bounds,
            dimension=dimension,
            **algorithm_params
        )
        
        # Run algorithm
        best_solution, best_fitness = algorithm.run()
        
        # Collect statistics
        stats = algorithm.get_statistics()
        
        run_result = {
            'best_fitness': best_fitness,
            'mean_fitness': stats['mean_fitness'],
            'std_fitness': stats['std_fitness'],
            'worst_fitness': stats['worst_fitness'],
            'evaluations': stats['evaluations'],
            'best_solution': best_solution,
            'parameters': algorithm_params
        }
        
        run_results.append(run_result)
        fitness_values.append(best_fitness)
        
        print(f"Best fitness: {best_fitness:.6f}")
    
    # Calculate overall statistics
    mean_fitness = np.mean(fitness_values)
    std_fitness = np.std(fitness_values)
    best_fitness = np.min(fitness_values)
    worst_fitness = np.max(fitness_values)
    
    # Calculate success rate (within 1% of global minimum)
    global_min = benchmark_func.global_minimum
    tolerance = abs(global_min) * 0.01 if global_min != 0 else 0.01
    success_count = sum(1 for f in fitness_values if f <= global_min + tolerance)
    success_rate = success_count / num_runs
    
    print(f"\n{'-'*60}")
    print(f"FINAL RESULTS for {algorithm_name} on {benchmark_func.name}:")
    print(f"Mean fitness: {mean_fitness:.6f}")
    print(f"Std fitness:  {std_fitness:.6f}")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Worst fitness: {worst_fitness:.6f}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"{'-'*60}")
    
    return {
        'algorithm_name': algorithm_name,
        'function_name': benchmark_func.name,
        'function_category': benchmark_func.category,
        'dimension': dimension,
        'num_runs': num_runs,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'best_fitness': best_fitness,
        'worst_fitness': worst_fitness,
        'success_rate': success_rate,
        'run_results': run_results,
        'parameters': algorithm_params
    }


def run_comprehensive_experiments():
    """Run comprehensive experiments on all benchmark functions."""
    
    # Initialize results manager
    results_manager = ResultsManager("results")
    
    # Experiment configuration
    dimension = 30
    num_runs = 20
    
    # Algorithm configurations to test
    ga_configurations = [
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
            'name': 'GA-Tournament-WholeArithmetic',
            'class': ModularGeneticAlgorithm,
            'params': {
                'crossover_type': 'whole_arithmetic',
                'mutation_type': 'gaussian',
                'selection_type': 'tournament',
                'replacement_type': 'generational',
                'population_size': 50,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01
            }
        },
        {
            'name': 'GA-Roulette-Simple',
            'class': ModularGeneticAlgorithm,
            'params': {
                'crossover_type': 'simple',
                'mutation_type': 'gaussian',
                'selection_type': 'roulette',
                'replacement_type': 'generational',
                'population_size': 50,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01
            }
        },
        {
            'name': 'Adaptive-GA',
            'class': AdaptiveGeneticAlgorithm,
            'params': {
                'crossover_type': 'arithmetic',
                'mutation_type': 'gaussian',
                'selection_type': 'tournament',
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
        }
    ]
    
    # Get functions to test (n-dimensional only for now)
    test_functions = []
    
    # Create function wrappers from BENCHMARK_FUNCTIONS
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
    
    for name, info in BENCHMARK_FUNCTIONS.items():
        # Skip 2D-only functions for 30D experiments
        if info["dim_type"] == 2 or info["dim_type"] == 1:
            continue
            
        bounds = info["bounds"]
        if isinstance(bounds, tuple) and len(bounds) == 2:
            # For 30D experiments, we need to repeat the bounds for each dimension
            bounds_list = [bounds] * 30  # Use 30 as default dimension
        else:
            bounds_list = bounds
            
        func_wrapper = FunctionWrapper(name, info["func"], bounds_list, info["type"])
        test_functions.append(func_wrapper)
    
    # Store all results for comparison
    all_results = {}
    
    print("COMPREHENSIVE OPTIMIZATION EXPERIMENTS")
    print("="*80)
    print(f"Testing {len(test_functions)} functions with {len(ga_configurations)} algorithms")
    print(f"Dimension: {dimension}, Runs per experiment: {num_runs}")
    print("="*80)
    
    # Run experiments
    for benchmark_func in test_functions:
        print(f"\n\nTESTING FUNCTION: {benchmark_func.name} ({benchmark_func.category})")
        print(f"Global minimum: {benchmark_func.global_minimum}")
        print(f"Bounds: {benchmark_func.get_bounds_for_dimension(dimension)[0]}")
        
        function_results = {}
        
        for config in ga_configurations:
            try:
                result = run_algorithm_experiment(
                    config['class'],
                    config['name'],
                    benchmark_func,
                    dimension,
                    num_runs,
                    **config['params']
                )
                
                # Save individual experiment results
                results_manager.save_experiment_results(
                    "comprehensive_study",
                    config['name'],
                    benchmark_func.name,
                    result['run_results'],
                    {
                        'function_category': benchmark_func.category,
                        'dimension': dimension,
                        'global_minimum': benchmark_func.global_minimum,
                        'algorithm_parameters': config['params']
                    }
                )
                
                function_results[config['name']] = {
                    'mean_fitness': result['mean_fitness'],
                    'std_fitness': result['std_fitness'],
                    'best_fitness': result['best_fitness'],
                    'worst_fitness': result['worst_fitness'],
                    'success_rate': result['success_rate'],
                    'parameters': config['params']
                }
                
            except Exception as e:
                print(f"Error running {config['name']} on {benchmark_func.name}: {str(e)}")
                continue
        
        all_results[benchmark_func.name] = function_results
    
    # Save comparison results
    results_manager.save_comparison_results("comprehensive_study", all_results)
    
    # Generate summary report
    results_manager.generate_summary_report("comprehensive_study")
    
    # Print summary table
    print("\n\n" + "="*80)
    print("EXPERIMENT COMPLETED - SUMMARY TABLE")
    print("="*80)
    
    # Create and display summary table
    from genetic_algorithm_optimization.utils.results_manager import create_results_summary_table
    create_results_summary_table("results")
    
    return all_results


def run_quick_demo():
    """Run a quick demo with fewer functions and runs for testing."""
    
    results_manager = ResultsManager("results")
    
    # Quick demo configuration
    dimension = 10
    num_runs = 3
    
    # Test a few key functions
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
            elif isinstance(self.bounds, list) and len(self.bounds) > 0:
                if isinstance(self.bounds[0], tuple):
                    return self.bounds
                else:
                    return [self.bounds[0]] * dim
            else:
                # Default bounds if none specified
                return [(-5.12, 5.12)] * dim
    
    # Get specific functions for demo
    demo_function_names = ["sphere", "ackley", "rastrigin", "rosenbrock"]
    for name in demo_function_names:
        if name in BENCHMARK_FUNCTIONS:
            info = BENCHMARK_FUNCTIONS[name]
            bounds = info["bounds"]
            # Create bounds list properly
            if isinstance(bounds, tuple) and len(bounds) == 2:
                bounds_list = bounds  # Keep as tuple for get_bounds_for_dimension to handle
            else:
                bounds_list = bounds
                
            func_wrapper = FunctionWrapper(name, info["func"], bounds_list, info["type"])
            demo_functions.append(func_wrapper)
    
    # Test a few key algorithms
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
            'name': 'PSO',
            'class': ParticleSwarmOptimization,
            'params': {
                'swarm_size': 30,
                'w': 0.74,
                'c1': 1.42,
                'c2': 1.42
            }
        }
    ]
    
    print("QUICK DEMO - OPTIMIZATION EXPERIMENTS")
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
                result = run_algorithm_experiment(
                    config['class'],
                    config['name'],
                    benchmark_func,
                    dimension,
                    num_runs,
                    **config['params']
                )
                
                function_results[config['name']] = {
                    'mean_fitness': result['mean_fitness'],
                    'std_fitness': result['std_fitness'],
                    'best_fitness': result['best_fitness'],
                    'worst_fitness': result['worst_fitness'],
                    'success_rate': result['success_rate']
                }
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
        all_results[benchmark_func.name] = function_results
    
    # Save results
    results_manager.save_comparison_results("quick_demo", all_results)
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("Running quick demo...")
        run_quick_demo()
    else:
        print("Running comprehensive experiments...")
        run_comprehensive_experiments()
