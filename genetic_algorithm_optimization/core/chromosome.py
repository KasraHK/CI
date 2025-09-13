import random
import copy


class Chromosome:
    """
    Chromosome class for Genetic Algorithm.
    
    Represents a solution as a list of genes (values for each dimension)
    with associated fitness and bounds information.
    """
    
    def __init__(self, genes, bounds, fitness=None):
        """
        Initialize a chromosome.
        
        Args:
            genes: List of values for each dimension
            bounds: List of tuples (min, max) for each dimension
            fitness: Fitness value (None if not evaluated)
        """
        self.genes = genes.copy() if genes else []
        self.bounds = bounds
        self.fitness = fitness
        self.dimension = len(genes) if genes else 0
    
    def copy(self):
        """Create a deep copy of the chromosome."""
        return Chromosome(self.genes, self.bounds, self.fitness)
    
    def __len__(self):
        """Return the number of genes (dimensions)."""
        return len(self.genes)
    
    def __getitem__(self, index):
        """Get gene value at given index."""
        return self.genes[index]
    
    def __setitem__(self, index, value):
        """Set gene value at given index."""
        self.genes[index] = value
    
    def __iter__(self):
        """Iterate over genes."""
        return iter(self.genes)
    
    def __str__(self):
        """String representation of the chromosome."""
        return f"Chromosome(genes={self.genes[:5]}{'...' if len(self.genes) > 5 else ''}, fitness={self.fitness})"
    
    def __repr__(self):
        """Detailed string representation."""
        return f"Chromosome(genes={self.genes}, bounds={self.bounds}, fitness={self.fitness})"
    
    def is_valid(self):
        """Check if all genes are within bounds."""
        for i, gene in enumerate(self.genes):
            min_val, max_val = self.bounds[i]
            if not (min_val <= gene <= max_val):
                return False
        return True
    
    def repair(self):
        """Repair genes that are outside bounds."""
        for i, gene in enumerate(self.genes):
            min_val, max_val = self.bounds[i]
            self.genes[i] = max(min_val, min(max_val, gene))
    
    def get_gene_range(self, index):
        """Get the valid range for a gene at given index."""
        return self.bounds[index]
    
    def set_fitness(self, fitness):
        """Set the fitness value."""
        self.fitness = fitness
    
    def get_fitness(self):
        """Get the fitness value."""
        return self.fitness
    
    def is_better_than(self, other):
        """Check if this chromosome is better than another (for minimization)."""
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness < other.fitness
    
    def distance_to(self, other):
        """Calculate Euclidean distance to another chromosome."""
        if len(self.genes) != len(other.genes):
            raise ValueError("Chromosomes must have same dimension")
        
        return sum((a - b) ** 2 for a, b in zip(self.genes, other.genes)) ** 0.5
    
    def hamming_distance_to(self, other):
        """Calculate Hamming distance to another chromosome (for discrete problems)."""
        if len(self.genes) != len(other.genes):
            raise ValueError("Chromosomes must have same dimension")
        
        return sum(1 for a, b in zip(self.genes, other.genes) if a != b)
    
    @classmethod
    def random(cls, bounds, dimension):
        """Create a random chromosome within given bounds."""
        genes = []
        for i in range(dimension):
            min_val, max_val = bounds[i]
            genes.append(random.uniform(min_val, max_val))
        return cls(genes, bounds)
    
    @classmethod
    def from_list(cls, gene_list, bounds):
        """Create chromosome from a list of genes."""
        return cls(gene_list, bounds)
