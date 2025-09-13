import random
import numpy as np
from .chromosome import Chromosome


class CrossoverOperator:
    """Base class for crossover operators."""
    
    def __init__(self, crossover_prob=0.75):
        """
        Initialize crossover operator.
        
        Args:
            crossover_prob: Probability of crossover
        """
        self.crossover_prob = crossover_prob
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            tuple: (child1, child2) chromosomes
        """
        raise NotImplementedError("Subclasses must implement crossover method")


class SimpleCrossover(CrossoverOperator):
    """
    Simple (single-point) crossover for continuous optimization.
    
    Selects a random crossover point and swaps genes after that point.
    """
    
    def crossover(self, parent1, parent2):
        """Perform simple crossover between two parents."""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        if len(parent1.genes) < 2:
            return parent1.copy(), parent2.copy()
        
        # Select random crossover point
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        
        # Create children
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
        
        child1 = Chromosome(child1_genes, parent1.bounds)
        child2 = Chromosome(child2_genes, parent2.bounds)
        
        # Ensure bounds are respected
        child1.repair()
        child2.repair()
        
        return child1, child2


class ArithmeticCrossover(CrossoverOperator):
    """
    Arithmetic crossover for continuous optimization.
    
    Creates children using linear combination: child = α * parent1 + (1-α) * parent2
    """
    
    def __init__(self, crossover_prob=0.75, alpha_range=(0.0, 1.0)):
        """
        Initialize arithmetic crossover.
        
        Args:
            crossover_prob: Probability of crossover
            alpha_range: Range for alpha parameter (min, max)
        """
        super().__init__(crossover_prob)
        self.alpha_range = alpha_range
    
    def crossover(self, parent1, parent2):
        """Perform arithmetic crossover between two parents."""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        # Generate random alpha
        alpha = random.uniform(*self.alpha_range)
        
        child1_genes = []
        child2_genes = []
        
        for i in range(len(parent1.genes)):
            # Arithmetic crossover: child = α * parent1 + (1-α) * parent2
            val1 = alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i]
            val2 = (1 - alpha) * parent1.genes[i] + alpha * parent2.genes[i]
            
            child1_genes.append(val1)
            child2_genes.append(val2)
        
        child1 = Chromosome(child1_genes, parent1.bounds)
        child2 = Chromosome(child2_genes, parent2.bounds)
        
        # Ensure bounds are respected
        child1.repair()
        child2.repair()
        
        return child1, child2


class WholeArithmeticCrossover(CrossoverOperator):
    """
    Whole arithmetic crossover for continuous optimization.
    
    Uses the same alpha value for all genes.
    """
    
    def __init__(self, crossover_prob=0.75, alpha_range=(0.0, 1.0)):
        """
        Initialize whole arithmetic crossover.
        
        Args:
            crossover_prob: Probability of crossover
            alpha_range: Range for alpha parameter (min, max)
        """
        super().__init__(crossover_prob)
        self.alpha_range = alpha_range
    
    def crossover(self, parent1, parent2):
        """Perform whole arithmetic crossover between two parents."""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        # Generate single random alpha for all genes
        alpha = random.uniform(*self.alpha_range)
        
        child1_genes = []
        child2_genes = []
        
        for i in range(len(parent1.genes)):
            # Whole arithmetic crossover: same alpha for all genes
            val1 = alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i]
            val2 = (1 - alpha) * parent1.genes[i] + alpha * parent2.genes[i]
            
            child1_genes.append(val1)
            child2_genes.append(val2)
        
        child1 = Chromosome(child1_genes, parent1.bounds)
        child2 = Chromosome(child2_genes, parent2.bounds)
        
        # Ensure bounds are respected
        child1.repair()
        child2.repair()
        
        return child1, child2


class OrderCrossover(CrossoverOperator):
    """
    Order crossover (OX) for permutation problems.
    
    Preserves relative order of elements from one parent.
    """
    
    def crossover(self, parent1, parent2):
        """Perform order crossover between two parents."""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        if len(parent1.genes) < 3:
            return parent1.copy(), parent2.copy()
        
        # Select two random crossover points
        start, end = sorted(random.sample(range(len(parent1.genes)), 2))
        
        # Create children
        child1_genes = [None] * len(parent1.genes)
        child2_genes = [None] * len(parent2.genes)
        
        # Copy segment from parent1 to child1
        child1_genes[start:end] = parent1.genes[start:end]
        
        # Fill remaining positions from parent2 in order
        remaining = [gene for gene in parent2.genes if gene not in child1_genes[start:end]]
        remaining_idx = 0
        
        for i in range(len(child1_genes)):
            if child1_genes[i] is None:
                child1_genes[i] = remaining[remaining_idx]
                remaining_idx += 1
        
        # Do the same for child2
        child2_genes[start:end] = parent2.genes[start:end]
        remaining = [gene for gene in parent1.genes if gene not in child2_genes[start:end]]
        remaining_idx = 0
        
        for i in range(len(child2_genes)):
            if child2_genes[i] is None:
                child2_genes[i] = remaining[remaining_idx]
                remaining_idx += 1
        
        child1 = Chromosome(child1_genes, parent1.bounds)
        child2 = Chromosome(child2_genes, parent2.bounds)
        
        return child1, child2


class CyclicCrossover(CrossoverOperator):
    """
    Cyclic crossover for permutation problems.
    
    Preserves cycles between parents.
    """
    
    def crossover(self, parent1, parent2):
        """Perform cyclic crossover between two parents."""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        if len(parent1.genes) < 2:
            return parent1.copy(), parent2.copy()
        
        # Find cycles
        cycles = []
        visited = [False] * len(parent1.genes)
        
        for i in range(len(parent1.genes)):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    # Find position of parent1[current] in parent2
                    current = parent2.genes.index(parent1.genes[current])
                cycles.append(cycle)
        
        # Create children
        child1_genes = [None] * len(parent1.genes)
        child2_genes = [None] * len(parent2.genes)
        
        for i, cycle in enumerate(cycles):
            if i % 2 == 0:  # Even cycles: child1 gets from parent1, child2 from parent2
                for pos in cycle:
                    child1_genes[pos] = parent1.genes[pos]
                    child2_genes[pos] = parent2.genes[pos]
            else:  # Odd cycles: child1 gets from parent2, child2 from parent1
                for pos in cycle:
                    child1_genes[pos] = parent2.genes[pos]
                    child2_genes[pos] = parent1.genes[pos]
        
        child1 = Chromosome(child1_genes, parent1.bounds)
        child2 = Chromosome(child2_genes, parent2.bounds)
        
        return child1, child2


class CrossoverFactory:
    """Factory class for creating crossover operators."""
    
    @staticmethod
    def create_crossover(crossover_type, **kwargs):
        """
        Create a crossover operator of specified type.
        
        Args:
            crossover_type: Type of crossover ('simple', 'arithmetic', 'whole_arithmetic', 'order', 'cyclic')
            **kwargs: Additional parameters for the crossover operator
            
        Returns:
            CrossoverOperator: Instance of the specified crossover operator
        """
        crossovers = {
            'simple': SimpleCrossover,
            'arithmetic': ArithmeticCrossover,
            'whole_arithmetic': WholeArithmeticCrossover,
            'order': OrderCrossover,
            'cyclic': CyclicCrossover
        }
        
        if crossover_type not in crossovers:
            raise ValueError(f"Unknown crossover type: {crossover_type}")
        
        return crossovers[crossover_type](**kwargs)
    
    @staticmethod
    def get_available_crossovers():
        """Get list of available crossover types."""
        return ['simple', 'arithmetic', 'whole_arithmetic', 'order', 'cyclic']
    
    @staticmethod
    def get_continuous_crossovers():
        """Get list of crossover types suitable for continuous optimization."""
        return ['simple', 'arithmetic', 'whole_arithmetic']
    
    @staticmethod
    def get_discrete_crossovers():
        """Get list of crossover types suitable for discrete/permutation problems."""
        return ['order', 'cyclic']
