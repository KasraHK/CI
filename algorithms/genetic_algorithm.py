import numpy as np
from typing import Callable, Tuple, List


import numpy as np
from typing import Callable, Tuple, List

class GeneticAlgorithm:
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: np.ndarray,
                 population_size: int = 50,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.75,
                 max_fitness_calls: int = 40000,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'single_point',
                 mutation_method: str = 'uniform'):
        
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_fitness_calls = max_fitness_calls
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        
        # Track fitness evaluations
        self.fitness_evaluations = 0
        
        # Initialize population
        self.initialize_population()
        
    def evaluate_fitness(self, individuals):
        """Evaluate fitness and track the number of evaluations
        
        Args:
            individuals (numpy.ndarray): Individual(s) to evaluate
            
        Returns:
            numpy.ndarray or float: Fitness value(s) for the individual(s)
        """
        if individuals.ndim == 1:
            fitness_value = self.objective_func(individuals)
            # Ensure we get a scalar value
            if hasattr(fitness_value, '__len__') and len(fitness_value) > 1:
                fitness_value = np.sum(fitness_value)
            self.fitness_evaluations += 1
            return fitness_value
        
        # For multiple individuals
        fitness_values = []
        for ind in individuals:
            fitness_value = self.objective_func(ind)
            # Ensure we get a scalar value
            if hasattr(fitness_value, '__len__') and len(fitness_value) > 1:
                fitness_value = np.sum(fitness_value)
            fitness_values.append(fitness_value)
            self.fitness_evaluations += 1
        
        return np.array(fitness_values)
        
    def initialize_population(self):
        """Initialize the population with random individuals within bounds"""
        self.population = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.population_size, self.dim)
        )
        self.fitness = self.evaluate_fitness(self.population)
        self.best_fitness_history = [np.min(self.fitness)]
        
    def tournament_selection(self, tournament_size=3):
        """Perform tournament selection
        
        Args:
            tournament_size (int): Number of individuals in each tournament
            
        Returns:
            numpy.ndarray: Selected individuals from the population
        """
        selected_indices = []
        for _ in range(self.population_size):
            contestants = np.random.choice(self.population_size, tournament_size, replace=False)
            winner = contestants[np.argmin(self.fitness[contestants])]
            selected_indices.append(winner)
        return self.population[selected_indices]
    
    def roulette_selection(self):
        """Perform roulette wheel (fitness proportionate) selection
        
        Returns:
            numpy.ndarray: Selected individuals from the population
        """
        # Fitness proportionate selection
        max_fitness = np.max(self.fitness)
        fitness_normalized = max_fitness - self.fitness + 1e-10  # Avoid division by zero
        probabilities = fitness_normalized / np.sum(fitness_normalized)
        selected_indices = np.random.choice(self.population_size, self.population_size, p=probabilities)
        return self.population[selected_indices]
    
    def selection(self):
        """Perform selection based on the configured selection method
        
        Returns:
            numpy.ndarray: Selected individuals from the population
        """
        if self.selection_method == 'tournament':
            return self.tournament_selection()
        elif self.selection_method == 'roulette':
            return self.roulette_selection()
        else:
            return self.tournament_selection()  # Default
    
    def single_point_crossover(self, parents):
        """Perform single-point crossover on parent pairs
        
        Args:
            parents (numpy.ndarray): Parent individuals
            
        Returns:
            numpy.ndarray: Offspring individuals
        """
        offspring = np.zeros_like(parents)
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size and np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                offspring[i] = np.concatenate([parents[i][:crossover_point], 
                                              parents[i+1][crossover_point:]])
                offspring[i+1] = np.concatenate([parents[i+1][:crossover_point], 
                                                parents[i][crossover_point:]])
            else:
                offspring[i] = parents[i]
                if i + 1 < self.population_size:
                    offspring[i+1] = parents[i+1]
        return offspring
    
    def uniform_crossover(self, parents):
        offspring = np.zeros_like(parents)
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size and np.random.rand() < self.crossover_rate:
                mask = np.random.rand(self.dim) < 0.5
                offspring[i] = np.where(mask, parents[i], parents[i+1])
                offspring[i+1] = np.where(mask, parents[i+1], parents[i])
            else:
                offspring[i] = parents[i]
                if i + 1 < self.population_size:
                    offspring[i+1] = parents[i+1]
        return offspring
    
    def crossover(self, parents):
        if self.crossover_method == 'single_point':
            return self.single_point_crossover(parents)
        elif self.crossover_method == 'uniform':
            return self.uniform_crossover(parents)
        else:
            return self.single_point_crossover(parents)  # Default
    
    def uniform_mutation(self, offspring):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(self.dim)
                offspring[i, mutation_point] = np.random.uniform(
                    self.bounds[mutation_point, 0],
                    self.bounds[mutation_point, 1]
                )
        return offspring
    
    def gaussian_mutation(self, offspring, sigma=0.1):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(self.dim)
                mutation = np.random.normal(0, sigma)
                offspring[i, mutation_point] = np.clip(
                    offspring[i, mutation_point] + mutation,
                    self.bounds[mutation_point, 0],
                    self.bounds[mutation_point, 1]
                )
        return offspring
    
    def mutation(self, offspring):
        if self.mutation_method == 'uniform':
            return self.uniform_mutation(offspring)
        elif self.mutation_method == 'gaussian':
            return self.gaussian_mutation(offspring)
        else:
            return self.uniform_mutation(offspring)  # Default
    
    def run(self):
        """Run the genetic algorithm optimization
        
        Returns:
            tuple: (best_fitness_history, best_position) - convergence history and best solution found
        """
        self.initialize_population()
        iteration = 0
        
        while self.fitness_evaluations < self.max_fitness_calls:
            # Selection
            parents = self.selection()
            # print(self.fitness_evaluations)
            
            # Crossover
            offspring = self.crossover(parents)
            
            # Mutation
            offspring = self.mutation(offspring)
            
            # Evaluate offspring
            offspring_fitness = self.evaluate_fitness(offspring)
            
            # Replace population (elitism)
            combined_population = np.vstack((self.population, offspring))
            combined_fitness = np.hstack((self.fitness, offspring_fitness))
            
            # Select best individuals
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            self.population = combined_population[best_indices]
            self.fitness = combined_fitness[best_indices]
            
            # Record best fitness
            self.best_fitness_history.append(np.min(self.fitness))
            
            iteration += 1
            
            # Early stopping if converged (disabled to use full fitness budget)
            # if len(self.best_fitness_history) > 50:
            #     recent_improvement = self.best_fitness_history[-50] - self.best_fitness_history[-1]
            #     if recent_improvement < 1e-10:
            #         break
            
            # Break if we've reached the maximum fitness evaluations
            if self.fitness_evaluations >= self.max_fitness_calls:
                break
        
        best_index = np.argmin(self.fitness)
        return self.best_fitness_history, self.population[best_index]
    