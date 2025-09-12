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

# ex = ExperimentRunner()
# ex.run_single_experiment('ackley2', 'ga', dim=10)
print(getattr(benchmarkfcns, 'ackley'))
# # Helper functions
# def get_all_function_names():
#     benchmark = BenchmarkSuite()
#     return benchmark.get_all_functions()

# def get_unimodal_functions():
#     benchmark = BenchmarkSuite()
#     return benchmark.get_unimodal_functions()

# def get_multimodal_functions():
#     benchmark = BenchmarkSuite()
#     return benchmark.get_multimodal_functions()

# def get_function_info():
#     benchmark = BenchmarkSuite()
#     return [benchmark.get_function_info(f) for f in benchmark.get_all_functions()]

# if __name__ == "__main__":
#     runner = ExperimentRunner()
    
#     # Get all available functions
#     all_functions = get_all_function_names()
#     unimodal_funcs = get_unimodal_functions()
#     multimodal_funcs = get_multimodal_functions()
    
#     print(f"Total functions: {len(all_functions)}")
#     print(f"Unimodal functions: {len(unimodal_funcs)}")
#     print(f"Multimodal functions: {len(multimodal_funcs)}")
    
#     # Run experiments for all functions
#     all_results, execution_times, fitness_calls = runner.run_all_experiments(
#         functions=all_functions,
#         algorithms=["ga", "pso"],
#         runs=30
#     )
    
#     print("Experiments completed successfully!")

