"""
Parallel optimization algorithms implementation.

This module contains parallel implementations of optimization algorithms
with parallel fitness evaluation capabilities for improved performance.
"""

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import partial
import time
from ..sequential.algorithms import ModularGeneticAlgorithm


def _evaluate_with_numpy_array(args):
    """Global function for parallel evaluation with numpy array conversion."""
    genes, objective_function = args
    genes_array = np.array(genes)
    return objective_function(genes_array)


class ParallelEvaluator:
    """
    Parallel evaluator for fitness functions.
    
    This class handles parallel evaluation of fitness functions across
    multiple processes, significantly speeding up optimization algorithms.
    
    Attributes:
        num_processes: Number of processes to use
        chunk_size: Number of tasks per process for load balancing
        pool: Multiprocessing pool instance
    """
    
    def __init__(self, num_processes=None, chunk_size=1):
        """
        Initialize parallel evaluator.
        
        Args:
            num_processes: Number of processes to use (None = auto-detect)
            chunk_size: Number of tasks per process (for load balancing)
        """
        self.num_processes = num_processes or min(cpu_count(), 8)  # Cap at 8 for stability
        self.chunk_size = chunk_size
        self.pool = None
        
    def __enter__(self):
        """Context manager entry."""
        self.pool = Pool(processes=self.num_processes)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.pool:
            self.pool.close()
            self.pool.join()
    
    def evaluate_population(self, objective_function, population):
        """
        Evaluate fitness for an entire population in parallel.
        
        Args:
            objective_function: Function to evaluate
            population: List of individuals (chromosomes or lists)
            
        Returns:
            list: Fitness values for each individual
        """
        if not population:
            return []
        
        # Extract genes from chromosomes if needed
        individuals = []
        for individual in population:
            if hasattr(individual, 'genes'):
                individuals.append(individual.genes)
            else:
                individuals.append(individual)
        
        # Prepare arguments for parallel evaluation
        args_list = [(genes, objective_function) for genes in individuals]
        
        # Evaluate in parallel
        fitness_values = self.pool.map(
            _evaluate_with_numpy_array, 
            args_list,
            chunksize=self.chunk_size
        )
        
        return fitness_values
    
    def evaluate_batch(self, objective_function, individuals):
        """
        Evaluate a batch of individuals in parallel.
        
        Args:
            objective_function: Function to evaluate
            individuals: List of individuals to evaluate
            
        Returns:
            list: Fitness values for each individual
        """
        return self.evaluate_population(objective_function, individuals)


def parallel_evaluate_population(objective_function, population, num_processes=None):
    """
    Convenience function for parallel population evaluation.
    
    Args:
        objective_function: Function to evaluate
        population: List of individuals
        num_processes: Number of processes to use
        
    Returns:
        list: Fitness values for each individual
    """
    with ParallelEvaluator(num_processes) as evaluator:
        return evaluator.evaluate_population(objective_function, population)


def benchmark_parallel_vs_sequential(objective_function, population, num_runs=5):
    """
    Benchmark parallel vs sequential evaluation.
    
    Args:
        objective_function: Function to evaluate
        population: List of individuals
        num_runs: Number of benchmark runs
        
    Returns:
        dict: Performance comparison results
    """
    print("Benchmarking Parallel vs Sequential Evaluation...")
    print(f"Population size: {len(population)}")
    print(f"Number of processes: {min(cpu_count(), 8)}")
    print(f"Benchmark runs: {num_runs}")
    
    # Sequential evaluation
    sequential_times = []
    for _ in range(num_runs):
        start_time = time.time()
        fitness_values = [objective_function(ind.genes if hasattr(ind, 'genes') else ind) 
                         for ind in population]
        sequential_times.append(time.time() - start_time)
    
    # Parallel evaluation
    parallel_times = []
    for _ in range(num_runs):
        start_time = time.time()
        fitness_values = parallel_evaluate_population(objective_function, population)
        parallel_times.append(time.time() - start_time)
    
    # Calculate statistics
    seq_mean = np.mean(sequential_times)
    seq_std = np.std(sequential_times)
    par_mean = np.mean(parallel_times)
    par_std = np.std(parallel_times)
    
    speedup = seq_mean / par_mean
    efficiency = speedup / min(cpu_count(), 8)
    
    results = {
        'sequential_mean': seq_mean,
        'sequential_std': seq_std,
        'parallel_mean': par_mean,
        'parallel_std': par_std,
        'speedup': speedup,
        'efficiency': efficiency,
        'population_size': len(population),
        'num_processes': min(cpu_count(), 8)
    }
    
    print(f"\nResults:")
    print(f"Sequential: {seq_mean:.4f} ± {seq_std:.4f} seconds")
    print(f"Parallel:   {par_mean:.4f} ± {par_std:.4f} seconds")
    print(f"Speedup:    {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.2%}")
    
    return results


class ParallelGeneticAlgorithm:
    """
    Parallel version of the modular genetic algorithm.
    
    This class extends the modular GA with parallel fitness evaluation
    while keeping all other functionality identical.
    
    Attributes:
        ga: Base modular genetic algorithm instance
        num_processes: Number of processes for parallel evaluation
        evaluator: Parallel evaluator instance
    """
    
    def __init__(self, objective_function, bounds, dimension, population_size=50,
                 crossover_type='arithmetic', mutation_type='gaussian', 
                 selection_type='tournament', replacement_type='generational',
                 crossover_prob=0.75, mutation_prob=0.01, num_processes=None, **kwargs):
        """
        Initialize parallel genetic algorithm.
        
        Args:
            objective_function: Function to optimize (minimize)
            bounds: List of tuples (min, max) for each dimension
            dimension: Number of dimensions
            population_size: Size of the population
            crossover_type: Type of crossover
            mutation_type: Type of mutation
            selection_type: Type of selection
            replacement_type: Type of replacement
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability per gene
            num_processes: Number of processes for parallel evaluation
            **kwargs: Additional parameters for operators
        """
        # Initialize the base GA
        self.ga = ModularGeneticAlgorithm(
            objective_function, bounds, dimension, population_size,
            crossover_type, mutation_type, selection_type, replacement_type,
            crossover_prob, mutation_prob, **kwargs
        )
        
        # Set up parallel evaluation
        self.num_processes = num_processes or min(cpu_count(), 8)
        self.evaluator = ParallelEvaluator(self.num_processes)
        
        # Override the evaluation method
        self.ga._evaluate_population = self._parallel_evaluate_population
        self.ga._evaluate_offspring = self._parallel_evaluate_offspring
    
    def _parallel_evaluate_population(self):
        """Parallel version of population evaluation."""
        fitness_values = []
        
        with self.evaluator:
            fitness_values = self.evaluator.evaluate_population(
                self.ga.objective_function, self.ga.population
            )
        
        # Update fitness values and track evaluations
        for i, fitness in enumerate(fitness_values):
            self.ga.population[i].set_fitness(fitness)
            self.ga.function_evaluations += 1
            
            # Update best solution if needed
            if fitness < self.ga.best_fitness:
                self.ga.best_fitness = fitness
                self.ga.best_solution = self.ga.population[i].copy()
        
        return fitness_values
    
    def _parallel_evaluate_offspring(self, offspring):
        """Parallel version of offspring evaluation."""
        child_fitness_values = []
        
        with self.evaluator:
            child_fitness_values = self.evaluator.evaluate_population(
                self.ga.objective_function, offspring
            )
        
        # Update fitness values and track evaluations
        for i, fitness in enumerate(child_fitness_values):
            offspring[i].set_fitness(fitness)
            self.ga.function_evaluations += 1
            
            # Update best solution if needed
            if fitness < self.ga.best_fitness:
                self.ga.best_fitness = fitness
                self.ga.best_solution = offspring[i].copy()
        
        return child_fitness_values
    
    def run(self):
        """Run the parallel genetic algorithm."""
        return self.ga.run()
    
    def get_statistics(self):
        """Get current population statistics."""
        return self.ga.get_statistics()


def create_parallel_ga_configurations():
    """
    Create parallel GA configurations for testing.
    
    Returns:
        list: List of parallel GA configurations
    """
    return [
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
        },
        {
            'name': 'Parallel-GA-Roulette-Simple',
            'class': ParallelGeneticAlgorithm,
            'params': {
                'crossover_type': 'simple',
                'mutation_type': 'gaussian',
                'selection_type': 'roulette',
                'replacement_type': 'generational',
                'population_size': 50,
                'crossover_prob': 0.75,
                'mutation_prob': 0.01,
                'num_processes': 4
            }
        }
    ]
