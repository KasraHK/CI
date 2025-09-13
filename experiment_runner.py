#!/usr/bin/env python3
"""
Focused Benchmark Experiment Runner
Runs experiments on specified benchmark functions and generates rendered tables.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.pso import ParticleSwarmOptimization
from benchmarks.functions import BenchmarkSuite
from config import EXPERIMENT_CONFIG, GA_CONFIG, PSO_CONFIG, TABLE_CONFIG, FUNCTION_LISTS

class BenchmarkExperimentRunner:
    def __init__(self, results_dir="results"):
        self.benchmark = BenchmarkSuite()
        self.results_dir = Path(results_dir)
        self.tables_dir = self.results_dir / "tables"
        self.data_dir = self.results_dir / "data"
        
        # Ensure directories exist
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load functions from configuration
        self.all_benchmark_functions = FUNCTION_LISTS["all_functions"]
        self.unimodal_functions = FUNCTION_LISTS["unimodal_functions"]
        self.multimodal_functions = FUNCTION_LISTS["multimodal_functions"]
        
        # Filter to only available functions
        self.available_functions = [f for f in self.all_benchmark_functions 
                                  if f in self.benchmark.functions]
        
        # Filter unimodal and multimodal functions to only available ones
        self.unimodal_functions = [f for f in self.unimodal_functions 
                                 if f in self.benchmark.functions]
        self.multimodal_functions = [f for f in self.multimodal_functions 
                                   if f in self.benchmark.functions]
        
    def run_single_experiment(self, func_name, algorithm, runs=5, dim=10):
        """Run a single experiment for one function and algorithm"""
        func_info = self.benchmark.get_function(func_name)
        
        # Check if the function supports the requested dimension
        required_dim = self.benchmark.get_required_dimension(func_name)
        if required_dim is not None:
            # Function has specific dimension requirement
            if dim != required_dim:
                print(f"⚠️  {func_name} only supports {required_dim}D input, adjusting dimension from {dim}D to {required_dim}D")
                dim = required_dim
        else:
            # Function supports any dimension, use the requested dimension
            pass
        
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
                    population_size=GA_CONFIG["population_size"],
                    mutation_rate=GA_CONFIG["mutation_rate"],
                    crossover_rate=GA_CONFIG["crossover_rate"],
                    max_fitness_calls=GA_CONFIG["max_fitness_calls"])
            elif algorithm == "pso":
                optimizer = ParticleSwarmOptimization(
                    objective_func=lambda x: self.benchmark.evaluate(func_name, x),
                    dim=dim,
                    bounds=bounds,
                    num_particles=PSO_CONFIG["num_particles"],
                    max_fitness_calls=PSO_CONFIG["max_fitness_calls"],
                    w=PSO_CONFIG["w"],
                    c1=PSO_CONFIG["c1"],
                    c2=PSO_CONFIG["c2"]
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
            best_scores, best_position = optimizer.run()
            end_time = time.time()
            
            results.append({
                "best_score": best_scores[-1],
                "best_position": best_position.tolist(),
                "convergence": best_scores,
                "run_time": end_time - start_time,
                "fitness_calls": optimizer.fitness_evaluations
            })
            
            run_times.append(end_time - start_time)
            fitness_calls_history.append(optimizer.fitness_evaluations)
            
        return results, np.mean(run_times), np.mean(fitness_calls_history)
    
    def run_category_experiments(self, functions, category_name, algorithms=["ga", "pso"], runs=10, dim=10):
        """Run experiments for a specific category of functions"""
        
        results = {}
        execution_times = {}
        fitness_calls = {}
        
        for func_name in tqdm(functions, desc=f"{category_name} Functions", ncols=80):
            if func_name not in self.benchmark.functions:
                print(f"❌ Skipping {func_name} - not implemented")
                continue
                
            results[func_name] = {}
            execution_times[func_name] = {}
            fitness_calls[func_name] = {}
            
            for algorithm in tqdm(algorithms, desc="Algorithms", leave=False, ncols=60):
                try:
                    exp_results, avg_time, avg_fitness_calls = self.run_single_experiment(
                        func_name, algorithm, runs, dim)
                    results[func_name][algorithm] = exp_results
                    execution_times[func_name][algorithm] = avg_time
                    fitness_calls[func_name][algorithm] = avg_fitness_calls
                except Exception as e:
                    print(f"❌ Error running {algorithm} on {func_name}: {e}")
                    results[func_name][algorithm] = []
                    execution_times[func_name][algorithm] = 0
                    fitness_calls[func_name][algorithm] = 0
        
        return results, execution_times, fitness_calls
    
    def create_results_table(self, results, execution_times, fitness_calls, category_name):
        """Create a results table for a category of functions"""
        table_data = []
        
        for func_name in results:
            func_info = self.benchmark.get_function(func_name)
            
            for algorithm in ["ga", "pso"]:
                if algorithm in results[func_name] and results[func_name][algorithm]:
                    # Calculate statistics
                    scores = [run["best_score"] for run in results[func_name][algorithm]]
                    times = [run["run_time"] for run in results[func_name][algorithm]]
                    fitness_evals = [run["fitness_calls"] for run in results[func_name][algorithm]]
                    
                    # Get the actual dimension used in the experiment
                    actual_dim = self.benchmark.get_required_dimension(func_name)
                    if actual_dim is None:
                        actual_dim = func_info['dim']  # Use default dimension for n-dimensional functions
                    
                    table_data.append({
                        "Function": func_name,
                        "Algorithm": algorithm.upper(),
                        "Best Score": f"{np.min(scores):.{TABLE_CONFIG['decimal_places']}f}",
                        "Mean Score": f"{np.mean(scores):.{TABLE_CONFIG['decimal_places']}f}",
                        "Std Score": f"{np.std(scores):.{TABLE_CONFIG['decimal_places']}f}",
                        "Mean Time (s)": f"{np.mean(times):.{TABLE_CONFIG['time_decimal_places']}f}",
                        "Mean Fitness Calls": f"{np.mean(fitness_evals):.0f}",
                        "Dimension": actual_dim,
                        "Min Value": func_info['min_value']
                    })
        
        df = pd.DataFrame(table_data)
        return df
    
    def save_results(self, unimodal_results, multimodal_results, 
                    unimodal_times, multimodal_times,
                    unimodal_fitness, multimodal_fitness):
        """Save all results to files with proper organization"""
        
        # Create tables
        unimodal_table = self.create_results_table(unimodal_results, unimodal_times, unimodal_fitness, "Unimodal")
        multimodal_table = self.create_results_table(multimodal_results, multimodal_times, multimodal_fitness, "Multimodal")
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save tables as CSV
        unimodal_csv = self.tables_dir / f"unimodal_results_{timestamp}.csv"
        multimodal_csv = self.tables_dir / f"multimodal_results_{timestamp}.csv"
        
        unimodal_table.to_csv(unimodal_csv, index=False)
        multimodal_table.to_csv(multimodal_csv, index=False)
        
        # Save detailed results as JSON
        detailed_results = {
            "experiment_info": {
                "timestamp": timestamp,
                "total_functions": len(self.available_functions),
                "unimodal_functions": len(self.unimodal_functions),
                "multimodal_functions": len(self.multimodal_functions),
                "algorithms": ["ga", "pso"]
            },
            "unimodal_results": {
                "results": unimodal_results,
                "execution_times": unimodal_times,
                "fitness_calls": unimodal_fitness
            },
            "multimodal_results": {
                "results": multimodal_results,
                "execution_times": multimodal_times,
                "fitness_calls": multimodal_fitness
            }
        }
        
        detailed_json = self.data_dir / f"detailed_results_{timestamp}.json"
        with open(detailed_json, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        return unimodal_table, multimodal_table, unimodal_csv, multimodal_csv, detailed_json
    
    def run_all_experiments(self, runs=None, dim=None):
        """Run all experiments and generate tables"""
        # Use configuration defaults if not specified
        runs = runs or EXPERIMENT_CONFIG["runs_per_experiment"]
        dim = dim or EXPERIMENT_CONFIG["default_dimension"]
        
        start_time = time.time()
        
        # Run unimodal experiments
        unimodal_results, unimodal_times, unimodal_fitness = self.run_category_experiments(
            self.unimodal_functions, "Unimodal", 
            algorithms=EXPERIMENT_CONFIG["algorithms"], runs=runs, dim=dim)
        
        # Run multimodal experiments  
        multimodal_results, multimodal_times, multimodal_fitness = self.run_category_experiments(
            self.multimodal_functions, "Multimodal", 
            algorithms=EXPERIMENT_CONFIG["algorithms"], runs=runs, dim=dim)
        
        # Create and save tables
        unimodal_table, multimodal_table, unimodal_csv, multimodal_csv, detailed_json = self.save_results(
            unimodal_results, multimodal_results,
            unimodal_times, multimodal_times,
            unimodal_fitness, multimodal_fitness
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("✅ EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"📊 Unimodal functions table: {unimodal_csv}")
        print(f"📊 Multimodal functions table: {multimodal_csv}")
        print(f"📁 Detailed results: {detailed_json}")
        print(f"\n📈 Unimodal table shape: {unimodal_table.shape}")
        print(f"📈 Multimodal table shape: {multimodal_table.shape}")
        
        return unimodal_table, multimodal_table

if __name__ == "__main__":
    runner = BenchmarkExperimentRunner()
    
    # Run experiments with configurable parameters
    unimodal_table, multimodal_table = runner.run_all_experiments(runs=5, dim=10)
    
    print("\n🎯 Experiment completed successfully!")
    print("📁 Check the 'results' folder for generated tables and data files.")
