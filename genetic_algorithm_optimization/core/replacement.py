import random
import numpy as np
from .chromosome import Chromosome


class ReplacementOperator:
    """Base class for replacement operators."""
    
    def replace(self, population, children, fitness_values, child_fitness_values):
        """
        Replace individuals in population with children.
        
        Args:
            population: Current population
            children: New children to potentially add
            fitness_values: Current population fitness values
            child_fitness_values: Children fitness values
            
        Returns:
            tuple: (new_population, new_fitness_values)
        """
        raise NotImplementedError("Subclasses must implement replace method")


class GenerationalReplacement(ReplacementOperator):
    """
    Generational replacement.
    
    Replaces entire population with children (plus elitism).
    """
    
    def __init__(self, elitism_count=1):
        """
        Initialize generational replacement.
        
        Args:
            elitism_count: Number of best individuals to preserve
        """
        self.elitism_count = elitism_count
    
    def replace(self, population, children, fitness_values, child_fitness_values):
        """Replace population using generational replacement."""
        # Find best individuals for elitism
        if self.elitism_count > 0:
            indexed_fitness = list(enumerate(fitness_values))
            indexed_fitness.sort(key=lambda x: x[1])  # Sort by fitness (ascending for minimization)
            elite_indices = [idx for idx, _ in indexed_fitness[:self.elitism_count]]
            elite = [population[i] for i in elite_indices]
        else:
            elite = []
        
        # Create new population: elite + children
        new_population = elite + children
        new_fitness_values = [fitness_values[i] for i in elite_indices] + child_fitness_values
        
        # If we have too many individuals, keep the best ones
        if len(new_population) > len(population):
            # Sort by fitness and keep the best
            indexed_fitness = list(enumerate(new_fitness_values))
            indexed_fitness.sort(key=lambda x: x[1])
            keep_indices = [idx for idx, _ in indexed_fitness[:len(population)]]
            
            new_population = [new_population[i] for i in keep_indices]
            new_fitness_values = [new_fitness_values[i] for i in keep_indices]
        
        return new_population, new_fitness_values


class SteadyStateReplacement(ReplacementOperator):
    """
    Steady-state replacement.
    
    Replaces only a few individuals per generation.
    """
    
    def __init__(self, replacement_count=2, replacement_strategy='worst'):
        """
        Initialize steady-state replacement.
        
        Args:
            replacement_count: Number of individuals to replace
            replacement_strategy: Strategy for selecting individuals to replace
                                ('worst', 'random', 'oldest', 'conservative')
        """
        self.replacement_count = replacement_count
        self.replacement_strategy = replacement_strategy
        self.age = {}  # Track age of individuals
    
    def replace(self, population, children, fitness_values, child_fitness_values):
        """Replace population using steady-state replacement."""
        new_population = population.copy()
        new_fitness_values = fitness_values.copy()
        
        # Update age
        for i in range(len(new_population)):
            self.age[i] = self.age.get(i, 0) + 1
        
        # Select individuals to replace
        if self.replacement_strategy == 'worst':
            # Replace worst individuals
            indexed_fitness = list(enumerate(new_fitness_values))
            indexed_fitness.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (descending)
            replace_indices = [idx for idx, _ in indexed_fitness[:self.replacement_count]]
        
        elif self.replacement_strategy == 'random':
            # Replace random individuals
            replace_indices = random.sample(range(len(new_population)), 
                                          min(self.replacement_count, len(new_population)))
        
        elif self.replacement_strategy == 'oldest':
            # Replace oldest individuals
            indexed_age = list(enumerate(self.age.values()))
            indexed_age.sort(key=lambda x: x[1], reverse=True)  # Sort by age (descending)
            replace_indices = [idx for idx, _ in indexed_age[:self.replacement_count]]
        
        elif self.replacement_strategy == 'conservative':
            # Replace worst individuals, but keep the best one
            indexed_fitness = list(enumerate(new_fitness_values))
            indexed_fitness.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (descending)
            replace_indices = [idx for idx, _ in indexed_fitness[1:self.replacement_count + 1]]
        
        else:
            raise ValueError(f"Unknown replacement strategy: {self.replacement_strategy}")
        
        # Replace selected individuals with children
        for i, replace_idx in enumerate(replace_indices):
            if i < len(children):
                new_population[replace_idx] = children[i]
                new_fitness_values[replace_idx] = child_fitness_values[i]
                self.age[replace_idx] = 0  # Reset age for new individual
        
        return new_population, new_fitness_values


class RandomReplacement(ReplacementOperator):
    """
    Random replacement.
    
    Randomly selects individuals to replace from a subset of parents.
    """
    
    def __init__(self, replacement_count=2, parent_subset_ratio=0.5, exclude_best=True):
        """
        Initialize random replacement.
        
        Args:
            replacement_count: Number of individuals to replace
            parent_subset_ratio: Fraction of parents to consider for replacement
            exclude_best: Whether to exclude the best individual from replacement
        """
        self.replacement_count = replacement_count
        self.parent_subset_ratio = parent_subset_ratio
        self.exclude_best = exclude_best
    
    def replace(self, population, children, fitness_values, child_fitness_values):
        """Replace population using random replacement."""
        new_population = population.copy()
        new_fitness_values = fitness_values.copy()
        
        # Find best individual if excluding
        if self.exclude_best:
            best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
        else:
            best_idx = None
        
        # Select subset of parents for potential replacement
        subset_size = max(1, int(len(population) * self.parent_subset_ratio))
        available_indices = [i for i in range(len(population)) if i != best_idx]
        subset_indices = random.sample(available_indices, min(subset_size, len(available_indices)))
        
        # Select individuals to replace from subset
        replace_indices = random.sample(subset_indices, 
                                      min(self.replacement_count, len(subset_indices)))
        
        # Replace selected individuals with children
        for i, replace_idx in enumerate(replace_indices):
            if i < len(children):
                new_population[replace_idx] = children[i]
                new_fitness_values[replace_idx] = child_fitness_values[i]
        
        return new_population, new_fitness_values


class ReplacementFactory:
    """Factory class for creating replacement operators."""
    
    @staticmethod
    def create_replacement(replacement_type, **kwargs):
        """
        Create a replacement operator of specified type.
        
        Args:
            replacement_type: Type of replacement ('generational', 'steady_state', 'random')
            **kwargs: Additional parameters for the replacement operator
            
        Returns:
            ReplacementOperator: Instance of the specified replacement operator
        """
        replacements = {
            'generational': GenerationalReplacement,
            'steady_state': SteadyStateReplacement,
            'random': RandomReplacement
        }
        
        if replacement_type not in replacements:
            raise ValueError(f"Unknown replacement type: {replacement_type}")
        
        return replacements[replacement_type](**kwargs)
    
    @staticmethod
    def get_available_replacements():
        """Get list of available replacement types."""
        return ['generational', 'steady_state', 'random']
    
    @staticmethod
    def get_steady_state_strategies():
        """Get list of available steady-state replacement strategies."""
        return ['worst', 'random', 'oldest', 'conservative']
