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
    "mutation_rate": 0.2,
    "crossover_rate": 0.75,
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
    ],
    "one_dimensional_functions": [
        'forrester',
        'gramacylee'
    ],
    "two_dimensional_functions": [
        'ackleyn2',
        'ackleyn3',
        'adjiman',
        'bartelsconn',
        'beale',
        'bird',
        'bohachevskyn1',
        'bohachevskyn2',
        'booth',
        'brent',
        'bukinn6',
        'carromtable',
        'crossintray',
        'deckersaarts',
        'dropwave',
        'easom',
        'eggcrate',
        'elattarvidyasagardutta',
        'goldsteinprice',
        'himmelblau',
        'holdertable',
        'keane',
        'leon',
        'levin13',
        'matyas',
        'mccormick',
        'schaffern1',
        'schaffern2',
        'schaffern3',
        'schaffern4',
        'threehumpcamel'
    ],
    "three_dimensional_functions": [
        'wolfe'
    ],
    "n_dimensional_functions": [
        'ackley',
        'ackleyn4',
        'alpine1',
        'alpine2',
        'brown',
        'exponential',
        'griewank',
        'happycat',
        'periodic',
        'powellsum',
        'qing',
        'quartic',
        'rastrigin',
        'ridge',
        'rosenbrock',
        'salomon',
        'schwefel220',
        'schwefel221',
        'schwefel222',
        'schwefel223',
        'schwefel',
        'shubert3',
        'shubertn4',
        'shubert',
        'sphere',
        'styblinskitank',
        'sumsquares',
        'trid',
        'xinsheyang',
        'xinsheyangn2',
        'xinsheyangn3',
        'xinsheyangn4',
        'zakharov'
    ]
}

F_MIN_INFO = {
    # Multimodal
    "ackley": (0.0, "(0,...,0)"),
    "ackley_n3": (-200.0, "(0, 0)"),
    "ackley_n4": (0.0, "(0,...,0)"),
    "adjiman": (-2.02181, "(2, 0.10578)"),
    "alpine_n1": (0.0, "(0,...,0)"),
    "alpine_n2": (2.808**30, "(7.917,...,7.917)"),
    "bartels_conn": (1.0, "(0, 0)"),
    "beale": (0.0, "(3, 0.5)"),
    "bird": (-106.764537, "multiple points"),
    "bohachevsky_n2": (0.0, "(0, 0)"),
    "bukin_n6": (0.0, "(-10, 1)"),
    "carrom_table": (-24.1568155, "4 symmetric points"),
    "cross_in_tray": (-2.06261, "4 symmetric points"),
    "deckkers_aarts": (-24777.0, "(0, ±15)"),
    "easom": (-1.0, "(pi, pi)"),
    "egg_crate": (0.0, "(0, 0)"),
    "el_attar_vidyasagar_dutta": (1.712780354, "(3.40918683, -2.171433679)"),
    "forrester": (-6.02074, "(0.757)"),
    "goldstein_price": (3.0, "(0, -1)"),
    "gramacy_lee": (-0.869, "(0.6)"),
    "happy_cat": (0.0, "(-1,...,-1)"),
    "himmelblau": (0.0, "4 points"),
    "holder_table": (-19.2085, "4 symmetric points"),
    "keane": (0.6736675, "symmetric points"),
    "levi_n13": (0.0, "(1, 1)"),
    "mccormick": (-1.9133, "(-0.54719, -1.54719)"),
    "periodic": (0.9, "(0,...,0)"),
    "qing": (0.0, "x_i = ±√i"),
    "quartic": (0.0, "(0,...,0)"),
    "rastrigin": (0.0, "(0,...,0)"),
    "rosenbrock": (0.0, "(1,...,1)"),
    "salomon": (0.0, "(0,...,0)"),
    "schwefel": (0.0, "(420.9687,...,420.9687)"),
    "shubert": (-186.7309, "multiple points"),
    "shubert_3": (-29.6738, "multiple points"),
    "shubert_n4": (-25.7174, "multiple points"),
    "styblinski_tang": (-39.16599*30, "(-2.903534,...,-2.903534)"),
    "wolfe": (0.0, "(0, 0, 0)"),
    "xin_she_yang": (0.0, "(0,...,0)"),
    "xin_she_yang_n2": (0.0, "(0,...,0)"),
    "xin_she_yang_n4": (0.0, "(0,...,0)"),
    # Unimodal
    "ackley_n2": (-200.0, "(0, 0)"),
    "bohachevsky_n1": (0.0, "(0, 0)"),
    "booth": (0.0, "(1, 3)"),
    "brent": (0.0, "(-10, -10)"),
    "brown": (0.0, "(0,...,0)"),
    "drop_wave": (-1.0, "(0, 0)"),
    "exponential": (-1.0, "(0,...,0)"),
    "griewank": (0.0, "(0,...,0)"),
    "leon": (0.0, "(1, 1)"),
    "matyas": (0.0, "(0, 0)"),
    "powell_sum": (0.0, "(0,...,0)"),
    "ridge": (0.0, "(0,...,0)"),
    "schaffer_n1": (0.0, "(0, 0)"),
    "schaffer_n2": (0.0, "(0, 0)"),
    "schaffer_n3": (0.00156685, "(0, 1.25313)"),
    "schaffer_n4": (0.292579, "(0, 1.25313)"),
    "schwefel_2_20": (0.0, "(0,...,0)"),
    "schwefel_2_21": (0.0, "(0,...,0)"),
    "schwefel_2_22": (0.0, "(0,...,0)"),
    "schwefel_2_23": (0.0, "(0,...,0)"),
    "sphere": (0.0, "(0,...,0)"),
    "sum_squares": (0.0, "(0,...,0)"),
    "three_hump_camel": (0.0, "(0, 0)"),
    "trid": (-30*(30+4)*(30-1)/6, "x_i = i*(31-i)"),
    "xin_she_yang_n3": (-1.0, "(0,...,0)"),
    "zakharov": (0.0, "(0,...,0)"),
}

