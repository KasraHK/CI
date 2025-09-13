"""
Benchmark functions for optimization algorithms.
This module provides a comprehensive set of benchmark functions for testing
genetic algorithms and particle swarm optimization.
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class BenchmarkFunction:
    """Represents a benchmark function with its properties."""
    name: str
    function: Callable
    bounds: List[Tuple[float, float]]
    global_minimum: float
    category: str  # 'unimodal' or 'multimodal'
    description: str = ""
    
    def get_bounds_for_dimension(self, dim: int) -> List[Tuple[float, float]]:
        """Get bounds for a specific dimension."""
        if isinstance(self.bounds[0], tuple):
            return self.bounds
        else:
            return [self.bounds] * dim

# Import all functions from manual_functions.py
from .manual_functions import *

def get_functions_by_category(category: str) -> List[BenchmarkFunction]:
    """Get all functions of a specific category (unimodal or multimodal)."""
    functions = []
    
    for name, info in benchmark_functions.items():
        if info["type"] == category:
            # Convert bounds to proper format
            bounds = info["bounds"]
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # Simple bounds for all dimensions
                bounds_list = [bounds]
            else:
                # Already a list of bounds
                bounds_list = bounds
            
            # Get global minimum (most functions have 0, some have specific values)
            global_min = 0.0
            if name in ["schwefel_2_26", "schwefel"]:
                global_min = -418.9829 * len(bounds_list) if isinstance(bounds, tuple) else -418.9829
            elif name in ["rosenbrock"]:
                global_min = 0.0
            elif name in ["ackley"]:
                global_min = 0.0
            elif name in ["rastrigin"]:
                global_min = 0.0
            elif name in ["griewank"]:
                global_min = 0.0
            
            func = BenchmarkFunction(
                name=name,
                function=info["func"],
                bounds=bounds_list,
                global_minimum=global_min,
                category=info["type"],
                description=f"{name} function"
            )
            functions.append(func)
    
    return functions

def get_function_by_name(name: str) -> BenchmarkFunction:
    """Get a specific function by name."""
    if name not in benchmark_functions:
        raise ValueError(f"Function '{name}' not found")
    
    info = benchmark_functions[name]
    bounds = info["bounds"]
    if isinstance(bounds, tuple) and len(bounds) == 2:
        bounds_list = [bounds]
    else:
        bounds_list = bounds
    
    global_min = 0.0
    if name in ["schwefel_2_26", "schwefel"]:
        global_min = -418.9829 * len(bounds_list) if isinstance(bounds, tuple) else -418.9829
    
    return BenchmarkFunction(
        name=name,
        function=info["func"],
        bounds=bounds_list,
        global_minimum=global_min,
        category=info["type"],
        description=f"{name} function"
    )

def get_all_functions() -> List[BenchmarkFunction]:
    """Get all available benchmark functions."""
    return get_functions_by_category("unimodal") + get_functions_by_category("multimodal")