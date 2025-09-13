import random
import numpy as np
from .chromosome import Chromosome


class MutationOperator:
    """Base class for mutation operators."""
    
    def __init__(self, mutation_prob=0.01):
        """
        Initialize mutation operator.
        
        Args:
            mutation_prob: Probability of mutation per gene
        """
        self.mutation_prob = mutation_prob
    
    def mutate(self, chromosome):
        """
        Apply mutation to a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Chromosome: Mutated chromosome
        """
        raise NotImplementedError("Subclasses must implement mutate method")


class GaussianMutation(MutationOperator):
    """
    Gaussian mutation for continuous optimization.
    
    Adds normally distributed noise to each gene with given probability.
    """
    
    def __init__(self, mutation_prob=0.01, sigma_factor=0.1):
        """
        Initialize Gaussian mutation.
        
        Args:
            mutation_prob: Probability of mutation per gene
            sigma_factor: Standard deviation as fraction of gene range
        """
        super().__init__(mutation_prob)
        self.sigma_factor = sigma_factor
    
    def mutate(self, chromosome):
        """Apply Gaussian mutation to chromosome."""
        mutated = chromosome.copy()
        
        for i in range(len(mutated.genes)):
            if random.random() < self.mutation_prob:
                min_val, max_val = mutated.bounds[i]
                sigma = (max_val - min_val) * self.sigma_factor
                
                # Add Gaussian noise
                mutation = np.random.normal(0, sigma)
                mutated.genes[i] += mutation
                
                # Ensure bounds are respected
                mutated.genes[i] = max(min_val, min(max_val, mutated.genes[i]))
        
        return mutated


class SwapMutation(MutationOperator):
    """
    Swap mutation for discrete/permutation problems.
    
    Randomly swaps two genes in the chromosome.
    """
    
    def __init__(self, mutation_prob=0.01):
        super().__init__(mutation_prob)
    
    def mutate(self, chromosome):
        """Apply swap mutation to chromosome."""
        if len(chromosome.genes) < 2:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if random.random() < self.mutation_prob:
            # Select two different random positions
            pos1, pos2 = random.sample(range(len(mutated.genes)), 2)
            # Swap the genes
            mutated.genes[pos1], mutated.genes[pos2] = mutated.genes[pos2], mutated.genes[pos1]
        
        return mutated


class InsertMutation(MutationOperator):
    """
    Insert mutation for discrete/permutation problems.
    
    Moves a gene to a different random position.
    """
    
    def __init__(self, mutation_prob=0.01):
        super().__init__(mutation_prob)
    
    def mutate(self, chromosome):
        """Apply insert mutation to chromosome."""
        if len(chromosome.genes) < 2:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if random.random() < self.mutation_prob:
            # Select source and destination positions
            src_pos = random.randint(0, len(mutated.genes) - 1)
            dest_pos = random.randint(0, len(mutated.genes) - 1)
            
            # Move gene from source to destination
            gene = mutated.genes.pop(src_pos)
            mutated.genes.insert(dest_pos, gene)
        
        return mutated


class ScrambleMutation(MutationOperator):
    """
    Scramble mutation for discrete/permutation problems.
    
    Randomly shuffles a contiguous segment of genes.
    """
    
    def __init__(self, mutation_prob=0.01, segment_size_factor=0.3):
        """
        Initialize scramble mutation.
        
        Args:
            mutation_prob: Probability of mutation
            segment_size_factor: Size of segment as fraction of chromosome length
        """
        super().__init__(mutation_prob)
        self.segment_size_factor = segment_size_factor
    
    def mutate(self, chromosome):
        """Apply scramble mutation to chromosome."""
        if len(chromosome.genes) < 2:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if random.random() < self.mutation_prob:
            # Determine segment size and position
            segment_size = max(2, int(len(mutated.genes) * self.segment_size_factor))
            segment_size = min(segment_size, len(mutated.genes))
            
            start_pos = random.randint(0, len(mutated.genes) - segment_size)
            end_pos = start_pos + segment_size
            
            # Scramble the segment
            segment = mutated.genes[start_pos:end_pos]
            random.shuffle(segment)
            mutated.genes[start_pos:end_pos] = segment
        
        return mutated


class InversionMutation(MutationOperator):
    """
    Inversion mutation for discrete/permutation problems.
    
    Reverses the order of genes in a contiguous segment.
    """
    
    def __init__(self, mutation_prob=0.01, segment_size_factor=0.3):
        """
        Initialize inversion mutation.
        
        Args:
            mutation_prob: Probability of mutation
            segment_size_factor: Size of segment as fraction of chromosome length
        """
        super().__init__(mutation_prob)
        self.segment_size_factor = segment_size_factor
    
    def mutate(self, chromosome):
        """Apply inversion mutation to chromosome."""
        if len(chromosome.genes) < 2:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if random.random() < self.mutation_prob:
            # Determine segment size and position
            segment_size = max(2, int(len(mutated.genes) * self.segment_size_factor))
            segment_size = min(segment_size, len(mutated.genes))
            
            start_pos = random.randint(0, len(mutated.genes) - segment_size)
            end_pos = start_pos + segment_size
            
            # Invert the segment
            segment = mutated.genes[start_pos:end_pos]
            segment.reverse()
            mutated.genes[start_pos:end_pos] = segment
        
        return mutated


class UniformMutation(MutationOperator):
    """
    Uniform mutation for continuous optimization.
    
    Replaces a gene with a random value within bounds.
    """
    
    def __init__(self, mutation_prob=0.01):
        super().__init__(mutation_prob)
    
    def mutate(self, chromosome):
        """Apply uniform mutation to chromosome."""
        mutated = chromosome.copy()
        
        for i in range(len(mutated.genes)):
            if random.random() < self.mutation_prob:
                min_val, max_val = mutated.bounds[i]
                mutated.genes[i] = random.uniform(min_val, max_val)
        
        return mutated


class MutationFactory:
    """Factory class for creating mutation operators."""
    
    @staticmethod
    def create_mutation(mutation_type, **kwargs):
        """
        Create a mutation operator of specified type.
        
        Args:
            mutation_type: Type of mutation ('gaussian', 'swap', 'insert', 'scramble', 'inversion', 'uniform')
            **kwargs: Additional parameters for the mutation operator
            
        Returns:
            MutationOperator: Instance of the specified mutation operator
        """
        mutations = {
            'gaussian': GaussianMutation,
            'swap': SwapMutation,
            'insert': InsertMutation,
            'scramble': ScrambleMutation,
            'inversion': InversionMutation,
            'uniform': UniformMutation
        }
        
        if mutation_type not in mutations:
            raise ValueError(f"Unknown mutation type: {mutation_type}")
        
        return mutations[mutation_type](**kwargs)
    
    @staticmethod
    def get_available_mutations():
        """Get list of available mutation types."""
        return ['gaussian', 'swap', 'insert', 'scramble', 'inversion', 'uniform']
