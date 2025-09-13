import numpy as np
from typing import Callable, List

class ParticleSwarmOptimization:
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: np.ndarray,
                 num_particles: int = 50,
                 max_fitness_calls: int = 40000,
                 w: float = 0.729,  # inertia weight
                 c1: float = 1.49445,  # cognitive coefficient
                 c2: float = 1.49445,  # social coefficient
                 v_max: float = None):  # maximum velocity
        
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_fitness_calls = max_fitness_calls
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Track fitness evaluations
        self.fitness_evaluations = 0
        
        # Set v_max to 20% of the search space if not provided
        if v_max is None:
            self.v_max = 0.2 * (bounds[:, 1] - bounds[:, 0])
        else:
            self.v_max = v_max
            
        self.best_fitness_history = []
        
    def evaluate_fitness(self, positions):
        """Evaluate fitness and track the number of evaluations"""
        if positions.ndim == 1:
            fitness_value = self.objective_func(positions)
            # Ensure we get a scalar value
            if hasattr(fitness_value, '__len__') and len(fitness_value) > 1:
                fitness_value = np.sum(fitness_value)
            self.fitness_evaluations += 1
            return fitness_value
        
        # For multiple positions
        fitness_values = []
        for pos in positions:
            fitness_value = self.objective_func(pos)
            # Ensure we get a scalar value
            if hasattr(fitness_value, '__len__') and len(fitness_value) > 1:
                fitness_value = np.sum(fitness_value)
            fitness_values.append(fitness_value)
            self.fitness_evaluations += 1
        
        return np.array(fitness_values)
        
    def initialize_particles(self):
        self.positions = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.num_particles, self.dim)
        )
        
        # Initialize velocities randomly within v_max bounds
        self.velocities = np.random.uniform(
            low=-self.v_max,
            high=self.v_max,
            size=(self.num_particles, self.dim)
        )
        
        # Evaluate initial positions
        fitness_values = self.evaluate_fitness(self.positions)
        
        # Initialize personal bests
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = fitness_values
        
        # Initialize global best
        self.global_best_index = np.argmin(self.personal_best_scores)
        self.global_best_position = np.copy(self.personal_best_positions[self.global_best_index])
        self.global_best_score = self.personal_best_scores[self.global_best_index]
        
    def update_velocities(self):
        r1 = np.random.random((self.num_particles, self.dim))
        r2 = np.random.random((self.num_particles, self.dim))
        
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)
        
        self.velocities = self.w * self.velocities + cognitive + social
        
        # Apply velocity limits
        for i in range(self.dim):
            self.velocities[:, i] = np.clip(
                self.velocities[:, i],
                -self.v_max[i],
                self.v_max[i]
            )
        
    def update_positions(self):
        self.positions += self.velocities
        
        # Apply bounds
        for i in range(self.dim):
            self.positions[:, i] = np.clip(
                self.positions[:, i], 
                self.bounds[i, 0], 
                self.bounds[i, 1]
            )
        
    def evaluate(self):
        scores = self.evaluate_fitness(self.positions)
        
        # Update personal best
        improved_indices = scores < self.personal_best_scores
        self.personal_best_positions[improved_indices] = self.positions[improved_indices]
        self.personal_best_scores[improved_indices] = scores[improved_indices]
        
        # Update global best
        if np.min(scores) < self.global_best_score:
            self.global_best_index = np.argmin(scores)
            self.global_best_position = np.copy(self.positions[self.global_best_index])
            self.global_best_score = scores[self.global_best_index]
            
    def run(self):
        self.initialize_particles()
        self.best_fitness_history.append(self.global_best_score)
        
        while self.fitness_evaluations < self.max_fitness_calls:
            self.update_velocities()
            self.update_positions()
            self.evaluate()
            
            self.best_fitness_history.append(self.global_best_score)
            
            # Early stopping if converged (disabled to use full fitness budget)
            # if len(self.best_fitness_history) > 50:
            #     recent_improvement = self.best_fitness_history[-50] - self.best_fitness_history[-1]
            #     if recent_improvement < 1e-10:
            #         break
            
            # Break if we've reached the maximum fitness evaluations
            if self.fitness_evaluations >= self.max_fitness_calls:
                break
            
        return self.best_fitness_history, self.global_best_position