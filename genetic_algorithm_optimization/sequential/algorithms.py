"""
Sequential optimization algorithms implementation.

This module contains the core sequential implementations of optimization algorithms
including Modular Genetic Algorithm, Adaptive Genetic Algorithm, and Particle Swarm Optimization.
"""

import random
import numpy as np
from ..core.chromosome import Chromosome
from ..core.mutations import MutationFactory
from ..core.crossovers import CrossoverFactory
from ..core.selection import SelectionFactory
from ..core.replacement import ReplacementFactory


class ModularGeneticAlgorithm:
    """
    Modular Genetic Algorithm implementation for continuous optimization problems.
    
    This class uses separate modules for mutations, crossovers, selection, and replacement,
    making it highly configurable and extensible.
    
    Attributes:
        objective_function: Function to optimize (minimize)
        bounds: List of tuples (min, max) for each dimension
        dimension: Number of dimensions
        population_size: Size of the population
        crossover_operator: Crossover operator instance
        mutation_operator: Mutation operator instance
        selection_operator: Selection operator instance
        replacement_operator: Replacement operator instance
        population: Current population of chromosomes
        fitness_values: Current fitness values
        best_solution: Best solution found so far
        best_fitness: Best fitness value found so far
        function_evaluations: Number of function evaluations performed
        max_evaluations: Maximum number of function evaluations allowed
    """
    
    def __init__(self, objective_function, bounds, dimension, population_size=50,
                 crossover_type='arithmetic', mutation_type='gaussian', 
                 selection_type='tournament', replacement_type='generational',
                 crossover_prob=0.75, mutation_prob=0.01, max_evaluations=40000, **kwargs):
        """
        Initialize the Modular Genetic Algorithm.
        
        Args:
            objective_function: Function to optimize (minimize)
            bounds: List of tuples (min, max) for each dimension
            dimension: Number of dimensions
            population_size: Size of the population (default: 50)
            crossover_type: Type of crossover ('simple', 'arithmetic', 'whole_arithmetic', 'order', 'cyclic')
            mutation_type: Type of mutation ('gaussian', 'swap', 'insert', 'scramble', 'inversion', 'uniform')
            selection_type: Type of selection ('roulette', 'tournament', 'rank', 'sus', 'truncation')
            replacement_type: Type of replacement ('generational', 'steady_state', 'random')
            crossover_prob: Crossover probability (default: 0.75)
            mutation_prob: Mutation probability per gene (default: 0.01)
            max_evaluations: Maximum function evaluations (default: 40000)
            **kwargs: Additional parameters for specific operators
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimension = dimension
        self.population_size = population_size
        
        # Track function evaluations
        self.function_evaluations = 0
        self.max_evaluations = max_evaluations
        
        # Track best solution
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Initialize operators with filtered parameters
        crossover_params = {k: v for k, v in kwargs.items() 
                           if k in ['alpha_range']}
        self.crossover_operator = CrossoverFactory.create_crossover(
            crossover_type, crossover_prob=crossover_prob, **crossover_params
        )
        
        mutation_params = {k: v for k, v in kwargs.items() 
                          if k in ['sigma_factor', 'segment_size_factor']}
        self.mutation_operator = MutationFactory.create_mutation(
            mutation_type, mutation_prob=mutation_prob, **mutation_params
        )
        
        selection_params = {k: v for k, v in kwargs.items() 
                           if k in ['tournament_size', 'selection_pressure']}
        self.selection_operator = SelectionFactory.create_selection(
            selection_type, **selection_params
        )
        
        replacement_params = {k: v for k, v in kwargs.items() 
                             if k in ['elitism_count', 'replacement_count', 'replacement_strategy', 
                                     'truncation_ratio', 'parent_subset_ratio', 'exclude_best']}
        self.replacement_operator = ReplacementFactory.create_replacement(
            replacement_type, **replacement_params
        )
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_values = self._evaluate_population()
        
        # Update best solution after initial evaluation
        self._update_best()
    
    def _initialize_population(self):
        """Initialize the population with random chromosomes within bounds."""
        population = []
        for _ in range(self.population_size):
            chromosome = Chromosome.random(self.bounds, self.dimension)
            population.append(chromosome)
        return population
    
    def _evaluate_population(self):
        """Evaluate fitness for all chromosomes in the population."""
        fitness_values = []
        for chromosome in self.population:
            # Ensure bounds are respected before evaluation
            if not chromosome.is_valid():
                chromosome.repair()
            
            # Convert genes to numpy array for compatibility with benchmark functions
            genes_array = np.array(chromosome.genes)
            fitness = self.objective_function(genes_array)
            chromosome.set_fitness(fitness)
            fitness_values.append(fitness)
            self.function_evaluations += 1
            
            # Update best solution if needed
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = chromosome.copy()
        
        return fitness_values
    
    def _update_best(self):
        """Update the best solution found so far."""
        for i, fitness in enumerate(self.fitness_values):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.population[i].copy()
    
    def _create_offspring(self):
        """Create offspring using selection, crossover, and mutation."""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Selection
            parents = self.selection_operator.select(
                self.population, self.fitness_values, num_parents=2
            )
            
            # Crossover
            children = self.crossover_operator.crossover(parents[0], parents[1])
            
            # Mutation
            for child in children:
                mutated_child = self.mutation_operator.mutate(child)
                offspring.append(mutated_child)
        
        # Trim to exact population size
        return offspring[:self.population_size]
    
    def _evaluate_offspring(self, offspring):
        """Evaluate fitness for offspring."""
        child_fitness_values = []
        for child in offspring:
            # Ensure bounds are respected before evaluation
            if not child.is_valid():
                child.repair()
            
            # Convert genes to numpy array for compatibility with benchmark functions
            genes_array = np.array(child.genes)
            fitness = self.objective_function(genes_array)
            child.set_fitness(fitness)
            child_fitness_values.append(fitness)
            self.function_evaluations += 1
            
            # Update best solution if needed
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = child.copy()
        
        return child_fitness_values
    
    def run(self):
        """
        Run the genetic algorithm until 40,000 function evaluations are reached.
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        generation = 0
        
        while self.function_evaluations < self.max_evaluations:
            # Create offspring
            offspring = self._create_offspring()
            
            # Evaluate offspring
            child_fitness_values = self._evaluate_offspring(offspring)
            
            # Replacement
            self.population, self.fitness_values = self.replacement_operator.replace(
                self.population, offspring, self.fitness_values, child_fitness_values
            )
            
            # Update best solution
            self._update_best()
            
            generation += 1
            
            # Print progress every 100 generations
            if generation % 100 == 0:
                print(f"Generation {generation}, Evaluations: {self.function_evaluations}, "
                      f"Best Fitness: {self.best_fitness:.6f}")
        
        print(f"GA completed after {generation} generations and {self.function_evaluations} evaluations")
        return self.best_solution, self.best_fitness
    
    def get_statistics(self):
        """Get current population statistics."""
        if not self.fitness_values:
            return {}
        
        return {
            'best_fitness': min(self.fitness_values),
            'worst_fitness': max(self.fitness_values),
            'mean_fitness': np.mean(self.fitness_values),
            'std_fitness': np.std(self.fitness_values),
            'evaluations': self.function_evaluations
        }
    
    def set_operators(self, crossover_type=None, mutation_type=None, 
                     selection_type=None, replacement_type=None, **kwargs):
        """
        Dynamically change operators during runtime.
        
        Args:
            crossover_type: New crossover type
            mutation_type: New mutation type
            selection_type: New selection type
            replacement_type: New replacement type
            **kwargs: Additional parameters for operators
        """
        if crossover_type:
            self.crossover_operator = CrossoverFactory.create_crossover(
                crossover_type, **kwargs
            )
        
        if mutation_type:
            self.mutation_operator = MutationFactory.create_mutation(
                mutation_type, **kwargs
            )
        
        if selection_type:
            self.selection_operator = SelectionFactory.create_selection(
                selection_type, **kwargs
            )
        
        if replacement_type:
            self.replacement_operator = ReplacementFactory.create_replacement(
                replacement_type, **kwargs
            )


class AdaptiveGeneticAlgorithm(ModularGeneticAlgorithm):
    """
    Adaptive Genetic Algorithm that changes operators based on generation or diversity.
    
    This class extends ModularGeneticAlgorithm with adaptive operator selection
    that changes based on the current generation and population diversity.
    
    Attributes:
        generation: Current generation number
        diversity_threshold: Threshold for diversity-based operator selection
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive genetic algorithm."""
        super().__init__(*args, **kwargs)
        self.generation = 0
        self.diversity_threshold = 0.1
    
    def _calculate_diversity(self):
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                total_distance += self.population[i].distance_to(self.population[j])
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _adaptive_operator_selection(self):
        """Select operators based on generation and diversity."""
        diversity = self._calculate_diversity()
        
        # Early generations: more exploration
        if self.generation < 200:
            if diversity < self.diversity_threshold:
                # Low diversity: use more explorative operators
                self.set_operators(
                    crossover_type='arithmetic',
                    mutation_type='gaussian',
                    selection_type='rank'
                )
            else:
                # Good diversity: use balanced operators
                self.set_operators(
                    crossover_type='simple',
                    mutation_type='gaussian',
                    selection_type='tournament'
                )
        else:
            # Later generations: more exploitation
            self.set_operators(
                crossover_type='whole_arithmetic',
                mutation_type='gaussian',
                selection_type='truncation'
            )
    
    def run(self):
        """Run adaptive genetic algorithm."""
        generation = 0
        
        while self.function_evaluations < self.max_evaluations:
            # Adaptive operator selection
            self._adaptive_operator_selection()
            
            # Create offspring
            offspring = self._create_offspring()
            
            # Evaluate offspring
            child_fitness_values = self._evaluate_offspring(offspring)
            
            # Replacement
            self.population, self.fitness_values = self.replacement_operator.replace(
                self.population, offspring, self.fitness_values, child_fitness_values
            )
            
            # Update best solution
            self._update_best()
            
            generation += 1
            self.generation = generation
            
            # Print progress every 100 generations
            if generation % 100 == 0:
                diversity = self._calculate_diversity()
                print(f"Generation {generation}, Evaluations: {self.function_evaluations}, "
                      f"Best Fitness: {self.best_fitness:.6f}, Diversity: {diversity:.6f}")
        
        print(f"Adaptive GA completed after {generation} generations and {self.function_evaluations} evaluations")
        return self.best_solution, self.best_fitness


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization implementation for continuous optimization problems.
    
    This class implements PSO with inertia weight, cognitive and social coefficients,
    and termination after exactly 40,000 function evaluations.
    
    Attributes:
        objective_function: Function to optimize (minimize)
        bounds: List of tuples (min, max) for each dimension
        dimension: Number of dimensions
        swarm_size: Size of the swarm
        w: Inertia weight
        c1: Cognitive coefficient
        c2: Social coefficient
        swarm: List of particles
        gbest_position: Global best position
        gbest_fitness: Global best fitness value
        function_evaluations: Number of function evaluations performed
        max_evaluations: Maximum number of function evaluations allowed
    """
    
    def __init__(self, objective_function, bounds, dimension, max_evaluations=40000, 
                 swarm_size=50, w=0.74, c1=1.42, c2=1.42):
        """
        Initialize the Particle Swarm Optimization algorithm.
        
        Args:
            objective_function: Function to optimize (minimize)
            bounds: List of tuples (min, max) for each dimension
            dimension: Number of dimensions
            max_evaluations: Maximum function evaluations (default: 40000)
            swarm_size: Size of the swarm (default: 50)
            w: Inertia weight (default: 0.74)
            c1: Cognitive coefficient (default: 1.42)
            c2: Social coefficient (default: 1.42)
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Track function evaluations
        self.function_evaluations = 0
        self.max_evaluations = max_evaluations
        
        # Initialize swarm
        self.swarm = [self._Particle(dimension, bounds) for _ in range(swarm_size)]
        
        # Global best position
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        
        # Evaluate initial swarm and find global best
        self._evaluate_swarm()
        self._update_global_best()
    
    class _Particle:
        """Particle class for Particle Swarm Optimization."""
        
        def __init__(self, dimension, bounds):
            """Initialize a particle."""
            self.dimension = dimension
            self.bounds = bounds
            
            # Initialize position randomly within bounds
            self.position = []
            for i in range(dimension):
                min_val, max_val = bounds[i]
                self.position.append(random.uniform(min_val, max_val))
            
            # Initialize velocity to zero
            self.velocity = [0.0] * dimension
            
            # Personal best position (initially same as current position)
            self.pbest_position = self.position.copy()
            self.pbest_fitness = float('inf')
    
    def _evaluate_swarm(self):
        """Evaluate fitness for all particles in the swarm."""
        for particle in self.swarm:
            # Ensure position is within bounds
            for i in range(len(particle.position)):
                min_val, max_val = self.bounds[i]
                particle.position[i] = max(min_val, min(max_val, particle.position[i]))
            
            # Convert position to numpy array for compatibility with benchmark functions
            position_array = np.array(particle.position)
            fitness = self.objective_function(position_array)
            self.function_evaluations += 1
            
            # Update personal best if current position is better
            if fitness < particle.pbest_fitness:
                particle.pbest_fitness = fitness
                particle.pbest_position = particle.position.copy()
    
    def _update_global_best(self):
        """Update the global best position."""
        for particle in self.swarm:
            if particle.pbest_fitness < self.gbest_fitness:
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_position = particle.pbest_position.copy()
    
    def _update_velocity(self, particle):
        """Update particle velocity using PSO velocity update equation."""
        for i in range(self.dimension):
            r1 = random.random()
            r2 = random.random()
            
            # PSO velocity update equation
            cognitive = self.c1 * r1 * (particle.pbest_position[i] - particle.position[i])
            social = self.c2 * r2 * (self.gbest_position[i] - particle.position[i])
            
            particle.velocity[i] = (self.w * particle.velocity[i] + 
                                  cognitive + social)
    
    def _update_position(self, particle):
        """Update particle position using current velocity."""
        for i in range(self.dimension):
            # Update position
            particle.position[i] += particle.velocity[i]
            
            # Apply bounds constraints
            min_val, max_val = self.bounds[i]
            particle.position[i] = max(min_val, min(max_val, particle.position[i]))
    
    def _update_particle(self, particle):
        """Update a single particle's velocity and position."""
        self._update_velocity(particle)
        self._update_position(particle)
    
    def run(self):
        """
        Run the particle swarm optimization until 40,000 function evaluations are reached.
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        iteration = 0
        
        while self.function_evaluations < self.max_evaluations:
            # Update all particles
            for particle in self.swarm:
                self._update_particle(particle)
            
            # Evaluate updated swarm
            self._evaluate_swarm()
            
            # Update global best
            self._update_global_best()
            
            iteration += 1
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Evaluations: {self.function_evaluations}, "
                      f"Best Fitness: {self.gbest_fitness:.6f}")
        
        print(f"PSO completed after {iteration} iterations and {self.function_evaluations} evaluations")
        return self.gbest_position, self.gbest_fitness
    
    def get_statistics(self):
        """
        Get statistics about the PSO run.
        
        Returns:
            dict: Dictionary containing statistics
        """
        return {
            'best_fitness': self.gbest_fitness,
            'best_solution': self.gbest_position,
            'function_evaluations': self.function_evaluations,
            'swarm_size': self.swarm_size
        }
