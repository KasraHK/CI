# 🧬 Genetic Algorithm Optimization Framework

A comprehensive, modular Python framework for genetic algorithms and particle swarm optimization with 67 benchmark functions.

## ✨ Features

- **67 Benchmark Functions** (26 unimodal + 41 multimodal)
- **Modular GA** with configurable operators
- **Particle Swarm Optimization (PSO)**
- **Multiple Selection Methods** (5 types)
- **Multiple Crossover Methods** (5 types) 
- **Multiple Mutation Methods** (6 types)
- **Multiple Replacement Strategies** (3 types)
- **Parallel Processing Support**
- **Comprehensive Statistics & Results**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd genetic_algorithm_optimization

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy

# Activate virtual environment (if using)
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows
```

### Basic Usage

```python
from genetic_algorithm_optimization.sequential.algorithms import ModularGeneticAlgorithm, ParticleSwarmOptimization
import numpy as np

# Define your objective function
def sphere_function(x):
    return np.sum(x**2)

# Set up problem
bounds = [(-5.12, 5.12) for _ in range(30)]  # 30D problem
dimension = 30

# Run GA
ga = ModularGeneticAlgorithm(
    objective_function=sphere_function,
    bounds=bounds,
    dimension=dimension,
    population_size=50,
    max_evaluations=40000
)

best_solution, best_fitness = ga.run()
print(f"Best fitness: {best_fitness}")

# Run PSO
pso = ParticleSwarmOptimization(
    objective_function=sphere_function,
    bounds=bounds,
    dimension=dimension,
    max_evaluations=40000,
    swarm_size=50
)

best_solution, best_fitness = pso.run()
print(f"Best fitness: {best_fitness}")
```

### Run Examples

```bash
# Run simple example
python simple_example.py

# Run main experiments
python main.py
```

## 📚 Documentation

For detailed documentation and tutorials, see [TUTORIAL.md](TUTORIAL.md).

## 🏗️ Project Structure

```
genetic_algorithm_optimization/
├── core/                    # Core components
│   ├── chromosome.py       # Chromosome class
│   ├── mutations.py        # Mutation operators
│   ├── crossovers.py       # Crossover operators
│   ├── selection.py        # Selection methods
│   └── replacement.py      # Replacement strategies
├── sequential/             # Sequential algorithms
│   └── algorithms.py       # GA and PSO implementations
├── parallel/               # Parallel processing
│   └── algorithms.py       # Parallel GA implementation
├── benchmarks/             # Benchmark functions
│   └── manual_functions.py # 67 benchmark functions
├── experiments/            # Experiment runners
│   ├── sequential_experiments.py
│   └── parallel_experiments.py
└── utils/                  # Utilities
    ├── results_manager.py
    └── visualize_results.py
```

## 🎯 Available Algorithms

### Genetic Algorithm (GA)
- **Selection Methods**: Tournament, Roulette Wheel, Rank, SUS, Truncation
- **Crossover Methods**: Arithmetic, Simple, Whole Arithmetic, Order, Cyclic
- **Mutation Methods**: Gaussian, Uniform, Swap, Insert, Scramble, Inversion
- **Replacement Strategies**: Generational, Steady-State, Random

### Particle Swarm Optimization (PSO)
- Configurable inertia weight (w)
- Cognitive coefficient (c1)
- Social coefficient (c2)
- Bounds checking and repair

## 📊 Benchmark Functions

### Unimodal Functions (26)
- Sphere, Ackley N.2, Bohachevsky N.1, Booth, Brent, Brown, Drop-Wave, Exponential, Griewank, Leon, Matyas, Powell Sum, Ridge, Schaffer N.1-4, Schwefel 2.20-23, Sum Squares, Three Hump Camel, Trid, Xinsheyang N.3, Zakharov

### Multimodal Functions (41)
- Ackley, Ackley N.3-4, Adjiman, Alpine N.1-2, Bartels Conn, Beale, Bird, Bohachevsky N.2, Bukin N.6, Carrom Table, Cross-in-Tray, Deckkers-Aarts, Easom, Egg Crate, Elattar-Vidyasagar-Dutta, Forrester, Goldstein-Price, Gramacy-Lee, Happy Cat, Himmelblau, Holder Table, Keane, Levi N.13, McCormick, Periodic, Qing, Quartic, Rastrigin, Rosenbrock, Salomon, Schwefel, Shubert, Shubert 3-4, Styblinski-Tang, Wolfe, Xinsheyang, Xinsheyang N.2-4

## 🔧 Configuration Examples

### Basic GA Configuration
```python
ga = ModularGeneticAlgorithm(
    objective_function=my_function,
    bounds=my_bounds,
    dimension=my_dimension,
    population_size=50,
    crossover_type='arithmetic',
    mutation_type='gaussian',
    selection_type='tournament',
    replacement_type='generational',
    crossover_prob=0.75,
    mutation_prob=0.01,
    max_evaluations=40000
)
```

### Advanced GA Configuration
```python
ga = ModularGeneticAlgorithm(
    objective_function=my_function,
    bounds=my_bounds,
    dimension=my_dimension,
    
    # Selection parameters
    selection_type='tournament',
    tournament_size=5,
    
    # Crossover parameters
    crossover_type='arithmetic',
    crossover_prob=0.8,
    alpha_range=(0.2, 0.8),
    
    # Mutation parameters
    mutation_type='gaussian',
    mutation_prob=0.02,
    sigma_factor=0.15,
    
    # Replacement parameters
    replacement_type='steady_state',
    replacement_count=4,
    replacement_strategy='conservative',
    
    # Population parameters
    population_size=100,
    max_evaluations=50000
)
```

### PSO Configuration
```python
pso = ParticleSwarmOptimization(
    objective_function=my_function,
    bounds=my_bounds,
    dimension=my_dimension,
    max_evaluations=40000,
    swarm_size=50,
    w=0.74,      # inertia weight
    c1=1.42,     # cognitive coefficient
    c2=1.42      # social coefficient
)
```

## 📈 Results

The framework automatically generates:
- **CSV files** with detailed results
- **Statistical analysis** (mean, std, min, max, success rate)
- **Convergence plots** (optional)
- **Performance comparisons**

## 🎓 Educational Use

This framework is perfect for:
- **Learning optimization algorithms**
- **Comparing different operators**
- **Research and experimentation**
- **Academic projects**
- **Algorithm benchmarking**

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you have any questions or issues, please open an issue on GitHub.

---

**🎉 Happy Optimizing!**