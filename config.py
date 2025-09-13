#!/usr/bin/env python3
"""
Configuration file for benchmark experiments
Modify these parameters to adjust experiment settings
"""

# Experiment Parameters
EXPERIMENT_CONFIG = {
    # Number of runs per function-algorithm combination
    "runs_per_experiment": 5,
    
    # Default dimension for functions that support any dimension
    "default_dimension": 10,
    
    # Algorithms to test
    "algorithms": ["ga", "pso"],
    
    # Results directory
    "results_directory": "results"
}

# Genetic Algorithm Parameters
GA_CONFIG = {
    "population_size": 50,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "max_fitness_calls": 40000
}

# Particle Swarm Optimization Parameters
PSO_CONFIG = {
    "num_particles": 100,
    "max_fitness_calls": 40000,
    "w": 0.9,  # Inertia weight
    "c1": 2.0,  # Cognitive parameter
    "c2": 2.0   # Social parameter
}

# Table Rendering Options
TABLE_CONFIG = {
    "decimal_places": 6,
    "time_decimal_places": 3,
    "include_timestamp": True,
    "save_csv": True,
    "save_json": True
}

# Function Lists
FUNCTION_LISTS = {
    "unimodal_functions": [
        'ackleyn2', 'bohachevskyn1', 'booth', 'brent', 'brown', 'dropwave', 'exponential',
        'powellsum', 'ridge', 'schaffern1', 'schaffern2', 'schaffern3', 'schaffern4',
        'schwefel220', 'schwefel221', 'schwefel222', 'schwefel223', 'sphere', 'sumsquares',
        'threehumpcamel', 'trid', 'xinsheyangn3'
    ],
    "multimodal_functions": [
        'ackley', 'ackleyn3', 'ackleyn4', 'adjiman', 'alpine1', 'alpine2', 'bartelsconn',
        'beale', 'bird', 'bohachevskyn2', 'bukinn6', 'carromtable', 'crossintray',
        'deckersaarts', 'easom', 'eggcrate', 'elattarvidyasagardutta', 'forrester',
        'goldsteinprice', 'gramacylee', 'griewank', 'happycat', 'himmelblau', 'holdertable',
        'keane', 'leon', 'levin13', 'matyas', 'mccormick', 'periodic', 'qing', 'quartic',
        'rastrigin', 'rosenbrock', 'salomon', 'schwefel', 'shubert3', 'shubertn4', 'shubert',
        'styblinskitank', 'wolfe', 'xinsheyang', 'xinsheyangn2', 'xinsheyangn4', 'zakharov'
    ],
    "all_functions": [
        # Unimodal functions
        'ackleyn2', 'bohachevskyn1', 'booth', 'brent', 'brown', 'dropwave', 'exponential',
        'powellsum', 'ridge', 'schaffern1', 'schaffern2', 'schaffern3', 'schaffern4',
        'schwefel220', 'schwefel221', 'schwefel222', 'schwefel223', 'sphere', 'sumsquares',
        'threehumpcamel', 'trid', 'xinsheyangn3',
        # Multimodal functions
        'ackley', 'ackleyn3', 'ackleyn4', 'adjiman', 'alpine1', 'alpine2', 'bartelsconn',
        'beale', 'bird', 'bohachevskyn2', 'bukinn6', 'carromtable', 'crossintray',
        'deckersaarts', 'easom', 'eggcrate', 'elattarvidyasagardutta', 'forrester',
        'goldsteinprice', 'gramacylee', 'griewank', 'happycat', 'himmelblau', 'holdertable',
        'keane', 'leon', 'levin13', 'matyas', 'mccormick', 'periodic', 'qing', 'quartic',
        'rastrigin', 'rosenbrock', 'salomon', 'schwefel', 'shubert3', 'shubertn4', 'shubert',
        'styblinskitank', 'wolfe', 'xinsheyang', 'xinsheyangn2', 'xinsheyangn4', 'zakharov'
    ]
}



