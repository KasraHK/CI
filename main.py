# from benchmarks.functions import cec2014_funcs
from benchmarks.functions import BenchmarkSuite
from experiments.run_experiments import ExperimentRunner
import benchmarkfcns
from benchmarks.functions import BenchmarkSuite
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.pso import ParticleSwarmOptimization
import numpy as np

# print(funcs['F25'])
# print(len(funcs))
# print(enumerate(cec2014_funcs.items())[2])

# bs = BenchmarkSuite()
# print(bs.get_function('rastrigin'))


# all_function_names = [
#     name for name in dir(benchmarkfcns) 
#     if not name.startswith('_') and callable(getattr(benchmarkfcns, name))
# ]
# for f in bs.get_multimodal_functions():
#     print(bs.get_function_info(f))

# from tqdm import tqdm
# import time

# for i in tqdm(range(300)):
#     time.sleep(0.005)
#     # print(i)
print(getattr(benchmarkfcns, 'bartelsconn'))
ex = ExperimentRunner()

# Test dimension handling
# print("Testing dimension compatibility:")
# print("=" * 50)

# # Test functions that support any dimension
# print("\n1. Testing sphere function (supports any dimension) with 10D:")
# print(ex.run_single_experiment('sphere', 'ga', dim=10, runs=10))

# # Test functions that only support 2D
# print("\n2. Testing eggcrate function (2D only) with 10D (should auto-adjust to 2D):")
# print(ex.run_single_experiment('eggcrate', 'ga', dim=10, runs=1))

# print("\n3. Testing eggcrate function (2D only) with 2D:")
# print(ex.run_single_experiment('eggcrate', 'ga', dim=2, runs=1))

# print("\n4. Testing ackleyn2 function (2D only) with 2D:")
# print(ex.run_single_experiment('ackleyn2', 'ga', dim=2, runs=1))

# # Show dimension compatibility info
# print("\n" + "=" * 50)
# print("Dimension compatibility information:")
# bs = BenchmarkSuite()
# dim_info = bs.get_functions_by_dimension_support()
# print(f"Functions supporting any dimension: {len(dim_info['any_dimension'])}")
# print(f"Functions requiring 2D only: {len(dim_info['two_dimension_only'])}")
# print(f"\nFirst 10 any-dimension functions: {dim_info['any_dimension'][:10]}")
# print(f"\nFirst 10 two-dimension-only functions: {dim_info['two_dimension_only'][:10]}")
# print(getattr(benchmarkfcns, 'ackley'))
# # Helper functions
def get_all_function_names():
    benchmark = BenchmarkSuite()
    return benchmark.get_all_functions()

def get_unimodal_functions():
    benchmark = BenchmarkSuite()
    return benchmark.get_unimodal_functions()

def get_multimodal_functions():
    benchmark = BenchmarkSuite()
    return benchmark.get_multimodal_functions()

def get_function_info():
    benchmark = BenchmarkSuite()
    return [benchmark.get_function_info(f) for f in benchmark.get_all_functions()]

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # Get all available functions
    all_functions = get_all_function_names()
    unimodal_funcs = get_unimodal_functions()
    multimodal_funcs = get_multimodal_functions()
    
    print(f"Total functions: {len(all_functions)}")
    print(f"Unimodal functions: {len(unimodal_funcs)}")
    print(f"Multimodal functions: {len(multimodal_funcs)}")
    
    # Run experiments for all functions
    all_results, execution_times, fitness_calls = runner.run_all_experiments(
        functions=all_functions,
        algorithms=["ga", "pso"],
        runs=30
    )
    
    print("Experiments completed successfully!")

