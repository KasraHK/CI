import random
import numpy as np
from .chromosome import Chromosome


class SelectionOperator:
    """Base class for selection operators."""
    
    def select(self, population, fitness_values, num_parents=2):
        """
        Select parents from population.
        
        Args:
            population: List of chromosomes
            fitness_values: List of fitness values
            num_parents: Number of parents to select
            
        Returns:
            list: Selected parent chromosomes
        """
        raise NotImplementedError("Subclasses must implement select method")


class RouletteWheelSelection(SelectionOperator):
    """
    Roulette wheel selection (fitness proportional selection).
    
    Probability of selection is proportional to fitness value.
    """
    
    def select(self, population, fitness_values, num_parents=2):
        """Select parents using roulette wheel selection."""
        # Convert fitness to selection probabilities (minimization problem)
        max_fitness = max(fitness_values)
        selection_probs = [max_fitness - fitness + 1e-10 for fitness in fitness_values]
        total_prob = sum(selection_probs)
        selection_probs = [prob / total_prob for prob in selection_probs]
        
        # Cumulative probabilities
        cumulative_probs = np.cumsum(selection_probs)
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            rand = random.random()
            for i, cum_prob in enumerate(cumulative_probs):
                if rand <= cum_prob:
                    parents.append(population[i].copy())
                    break
        
        return parents


class TournamentSelection(SelectionOperator):
    """
    Tournament selection.
    
    Randomly selects k individuals and picks the best one.
    """
    
    def __init__(self, tournament_size=3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Size of tournament (k)
        """
        self.tournament_size = tournament_size
    
    def select(self, population, fitness_values, num_parents=2):
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(num_parents):
            # Select tournament participants
            tournament_indices = random.sample(range(len(population)), 
                                            min(self.tournament_size, len(population)))
            
            # Find best individual in tournament
            best_idx = min(tournament_indices, key=lambda i: fitness_values[i])
            parents.append(population[best_idx].copy())
        
        return parents


class RankSelection(SelectionOperator):
    """
    Rank selection.
    
    Selection probability based on rank rather than raw fitness.
    """
    
    def __init__(self, selection_pressure=1.5):
        """
        Initialize rank selection.
        
        Args:
            selection_pressure: Selection pressure (1.0 = no pressure, 2.0 = high pressure)
        """
        self.selection_pressure = selection_pressure
    
    def select(self, population, fitness_values, num_parents=2):
        """Select parents using rank selection."""
        # Create list of (index, fitness) pairs and sort by fitness
        indexed_fitness = list(enumerate(fitness_values))
        indexed_fitness.sort(key=lambda x: x[1])  # Sort by fitness (ascending for minimization)
        
        # Assign ranks (1 = best, n = worst)
        ranks = [0] * len(population)
        for rank, (idx, _) in enumerate(indexed_fitness, 1):
            ranks[idx] = rank
        
        # Calculate selection probabilities based on ranks
        n = len(population)
        selection_probs = []
        for rank in ranks:
            # Linear ranking: P(i) = (2 - SP) / n + 2 * (SP - 1) * (n - rank + 1) / (n * (n - 1))
            prob = (2 - self.selection_pressure) / n + 2 * (self.selection_pressure - 1) * (n - rank + 1) / (n * (n - 1))
            selection_probs.append(prob)
        
        # Normalize probabilities
        total_prob = sum(selection_probs)
        selection_probs = [prob / total_prob for prob in selection_probs]
        
        # Cumulative probabilities
        cumulative_probs = np.cumsum(selection_probs)
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            rand = random.random()
            for i, cum_prob in enumerate(cumulative_probs):
                if rand <= cum_prob:
                    parents.append(population[i].copy())
                    break
        
        return parents


class StochasticUniversalSampling(SelectionOperator):
    """
    Stochastic Universal Sampling (SUS).
    
    More uniform selection than roulette wheel.
    """
    
    def select(self, population, fitness_values, num_parents=2):
        """Select parents using stochastic universal sampling."""
        # Convert fitness to selection probabilities (minimization problem)
        max_fitness = max(fitness_values)
        selection_probs = [max_fitness - fitness + 1e-10 for fitness in fitness_values]
        total_prob = sum(selection_probs)
        selection_probs = [prob / total_prob for prob in selection_probs]
        
        # Calculate cumulative probabilities
        cumulative_probs = np.cumsum(selection_probs)
        
        # SUS selection
        parents = []
        step = 1.0 / num_parents
        start = random.uniform(0, step)
        
        for i in range(num_parents):
            pointer = start + i * step
            for j, cum_prob in enumerate(cumulative_probs):
                if pointer <= cum_prob:
                    parents.append(population[j].copy())
                    break
        
        return parents


class TruncationSelection(SelectionOperator):
    """
    Truncation selection.
    
    Selects only from the top X% of the population.
    """
    
    def __init__(self, truncation_ratio=0.5):
        """
        Initialize truncation selection.
        
        Args:
            truncation_ratio: Fraction of population to consider (0.5 = top 50%)
        """
        self.truncation_ratio = truncation_ratio
    
    def select(self, population, fitness_values, num_parents=2):
        """Select parents using truncation selection."""
        # Sort population by fitness
        indexed_fitness = list(enumerate(fitness_values))
        indexed_fitness.sort(key=lambda x: x[1])  # Sort by fitness (ascending for minimization)
        
        # Select top individuals
        num_top = max(1, int(len(population) * self.truncation_ratio))
        top_indices = [idx for idx, _ in indexed_fitness[:num_top]]
        
        # Select parents from top individuals
        parents = []
        for _ in range(num_parents):
            idx = random.choice(top_indices)
            parents.append(population[idx].copy())
        
        return parents


class SelectionFactory:
    """Factory class for creating selection operators."""
    
    @staticmethod
    def create_selection(selection_type, **kwargs):
        """
        Create a selection operator of specified type.
        
        Args:
            selection_type: Type of selection ('roulette', 'tournament', 'rank', 'sus', 'truncation')
            **kwargs: Additional parameters for the selection operator
            
        Returns:
            SelectionOperator: Instance of the specified selection operator
        """
        selections = {
            'roulette': RouletteWheelSelection,
            'tournament': TournamentSelection,
            'rank': RankSelection,
            'sus': StochasticUniversalSampling,
            'truncation': TruncationSelection
        }
        
        if selection_type not in selections:
            raise ValueError(f"Unknown selection type: {selection_type}")
        
        return selections[selection_type](**kwargs)
    
    @staticmethod
    def get_available_selections():
        """Get list of available selection types."""
        return ['roulette', 'tournament', 'rank', 'sus', 'truncation']
    
    @staticmethod
    def get_recommended_selections():
        """Get list of commonly recommended selection methods."""
        return ['tournament', 'rank', 'sus']
