import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import time

from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.pso import ParticleSwarmOptimization
from benchmarks.functions import BenchmarkSuite

class ExperimentRunner:
    def __init__(self, results_dir="experiments/results"):
        self.benchmark = BenchmarkSuite()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_experiment(self, func_name, algorithm, runs=5, dim=10):
        func_info = self.benchmark.get_function(func_name)
        
        # Check if the function supports the requested dimension
        if not self.benchmark.is_dimension_compatible(func_name, dim):
            required_dim = self.benchmark.get_required_dimension(func_name)
            if required_dim is not None:
                print(f"Warning: {func_name} only supports {required_dim}D input, adjusting dimension from {dim}D to {required_dim}D")
                dim = required_dim
            else:
                print(f"Warning: {func_name} may not support {dim}D input, proceeding anyway...")
        
        # Get bounds for the adjusted dimension
        bounds = self.benchmark.get_reasonable_bounds(func_name, dim)
        
        results = []
        run_times = []
        fitness_calls_history = []
        
        for run in range(runs):
            start_time = time.time()
            
            if algorithm == "ga":
                optimizer = GeneticAlgorithm(
                    objective_func=lambda x: self.benchmark.evaluate(func_name, x),
                    dim=dim,
                    bounds=bounds,
                    population_size=50,
                    mutation_rate=0.1,
                    crossover_rate=0.8,
                    max_fitness_calls=40000)
            elif algorithm == "pso":
                optimizer = ParticleSwarmOptimization(
                    objective_func=lambda x: self.benchmark.evaluate(func_name, x),
                    dim=dim,
                    bounds=bounds,
                    num_particles=100,
                    max_fitness_calls=40000,
                    w=0.9,
                    c1=2.0,
                    c2=2.0
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
            best_scores, best_position = optimizer.run()
            end_time = time.time()
            
            results.append({
                "best_score": best_scores[-1],
                # "best_position": best_position.tolist(),
                # "convergence": best_scores,
                # "run_time": end_time - start_time,
                # "fitness_calls": optimizer.fitness_evaluations
            })
            
            run_times.append(end_time - start_time)
            fitness_calls_history.append(optimizer.fitness_evaluations)
            
        return results, np.mean(run_times), np.mean(fitness_calls_history)
    
    def run_all_experiments(self, functions, algorithms, runs=20, dim=10):
        all_results = {}
        execution_times = {}
        fitness_calls = {}
        
        for func_name in tqdm(functions, desc="Functions"):
            if func_name not in self.benchmark.functions:
                print(f"Skipping {func_name} - not implemented")
                continue
                
            all_results[func_name] = {}
            execution_times[func_name] = {}
            fitness_calls[func_name] = {}
            
            for algorithm in tqdm(algorithms, desc="Algorithms", leave=False):
                print(f"Running {algorithm} on {func_name}")
                results, avg_time, avg_fitness_calls = self.run_single_experiment(func_name, algorithm, runs, dim)
                all_results[func_name][algorithm] = results
                execution_times[func_name][algorithm] = avg_time
                fitness_calls[func_name][algorithm] = avg_fitness_calls
                
                # Save intermediate results
                with open(self.results_dir / f"{func_name}_{algorithm}.json", "w") as f:
                    json.dump({
                        "function": func_name,
                        "algorithm": algorithm,
                        "results": results,
                        "average_time": avg_time,
                        "average_fitness_calls": avg_fitness_calls
                    }, f, indent=2)
        
        # Save all results
        with open(self.results_dir / "all_results.json", "w") as f:
            json.dump({
                "all_results": all_results,
                "execution_times": execution_times,
                "fitness_calls": fitness_calls
            }, f, indent=2)
            
        return all_results, execution_times, fitness_calls

# Helper functions
def get_all_function_names():
    benchmark = BenchmarkSuite()
    return benchmark.get_all_functions()

def get_unimodal_functions():
    benchmark = BenchmarkSuite()
    return benchmark.get_unimodal_functions()

def get_multimodal_functions():
    benchmark = BenchmarkSuite()
    return benchmark.get_multimodal_functions()

def get_function_info():
    benchmark = BenchmarkSuite()
    return [benchmark.get_function_info(f) for f in benchmark.get_all_functions()]

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # Get all available functions
    all_functions = get_all_function_names()
    unimodal_funcs = get_unimodal_functions()
    multimodal_funcs = get_multimodal_functions()
    
    print(f"Total functions: {len(all_functions)}")
    print(f"Unimodal functions: {len(unimodal_funcs)}")
    print(f"Multimodal functions: {len(multimodal_funcs)}")
    
    # Run experiments for all functions
    all_results, execution_times, fitness_calls = runner.run_all_experiments(
        functions=all_functions,
        algorithms=["ga", "pso"],
        runs=30
    )
    
    print("Experiments completed successfully!")