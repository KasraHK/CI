# Benchmark Optimization Project

A focused benchmark experiment system for testing optimization algorithms on specific benchmark functions.

## Project Structure

```
CI/
├── algorithms/                 # Optimization algorithms
│   ├── genetic_algorithm.py   # Genetic Algorithm implementation
│   └── pso.py                 # Particle Swarm Optimization implementation
├── benchmarks/                # Benchmark functions
│   ├── __init__.py
│   └── functions.py           # Benchmark function suite
├── results/                   # Experiment results
│   ├── tables/               # Generated CSV tables
│   └── data/                 # Detailed JSON results
├── config.py                 # Configuration parameters
├── experiment_runner.py      # Main experiment runner
├── main.py                   # Entry point
├── Pipfile                   # Python dependencies
└── README.md                 # This file
```

## Configuration

Edit `config.py` to modify experiment parameters:

- **Experiment settings:** Number of runs, dimension, algorithms
- **Algorithm parameters:** GA and PSO settings
- **Table formatting:** Decimal places, file naming
- **Function lists:** Which functions to test

## Generated Tables

The system generates two main tables:

1. **Unimodal Functions Table** - Functions with single global optimum
2. **Multimodal Functions Table** - Functions with multiple local optima

Each table contains:
- Function name and algorithm
- Best, mean, and standard deviation of scores
- Execution time and fitness evaluations
- Function dimension and known minimum value

## Available Functions

**61 out of 67 specified functions** are available:

- **Unimodal (12):** powellsum, quartic, ridge, rosenbrock, schwefel220, schwefel221, schwefel222, schwefel223, sphere, sumsquares, trid, zakharov
- **Multimodal (49):** ackley, ackleyn2, ackleyn3, ackleyn4, adjiman, bartelsconn, beale, bird, bohachevskyn1, bohachevskyn2, booth, brent, brown, bukinn6, carromtable, crossintray, dropwave, easom, eggcrate, exponential, forrester, goldsteinprice, gramacylee, griewank, happycat, himmelblau, holdertable, keane, leon, levin13, matyas, mccormick, periodic, qing, rastrigin, salomon, schaffern1, schaffern2, schaffern3, schaffern4, schwefel, shubertn4, shubert, styblinskitank, threehumpcamel, wolfe, xinsheyangn2, xinsheyangn3, xinsheyangn4

**Missing functions (6):** alpine1, alpine2, deckersaarts, elattarvidyasagardutta, shubert3, xinsheyang

## Customization

### Changing Algorithm Parameters

Edit `config.py`:

```python
# Genetic Algorithm
GA_CONFIG = {
    "population_size": 50,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "max_fitness_calls": 40000
}

# Particle Swarm Optimization
PSO_CONFIG = {
    "num_particles": 100,
    "max_fitness_calls": 40000,
    "w": 0.9,  # Inertia weight
    "c1": 2.0,  # Cognitive parameter
    "c2": 2.0   # Social parameter
}
```

### Changing Experiment Settings

```python
EXPERIMENT_CONFIG = {
    "runs_per_experiment": 5,    # Number of runs per function
    "default_dimension": 10,     # Default dimension
    "algorithms": ["ga", "pso"], # Algorithms to test
    "results_directory": "results"
}
```

## Output Files

- **CSV Tables:** `results/tables/unimodal_results_TIMESTAMP.csv`
- **Detailed JSON:** `results/data/detailed_results_TIMESTAMP.json`
- **Console Output:** Real-time progress and summary

## Dependencies

- Python 3.7+
- NumPy
- Pandas
- tqdm
- benchmarkfcns

## Notes

- Functions requiring 2D input are automatically handled
- Dimension adjustments are logged with warnings
- Results are timestamped for easy organization
- The system is designed for your specific 67-function list



