"""
Benchmark functions for optimization testing.

This module provides a comprehensive collection of unimodal and multimodal
benchmark functions commonly used in optimization research.
"""

from .functions import (
    BenchmarkFunction,
    get_functions_by_category,
    get_function_by_name,
    get_all_functions
)

from .manual_functions import BENCHMARK_FUNCTIONS as benchmark_functions

__all__ = [
    "BenchmarkFunction",
    "get_functions_by_category",
    "get_function_by_name",
    "get_all_functions",
    "benchmark_functions"
]
