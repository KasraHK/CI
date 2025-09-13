#!/usr/bin/env python3
"""
Quick test of the experiment with a few functions to verify global minimums are displayed
"""

from final_standard_experiment import run_final_standard_experiment
from genetic_algorithm_optimization.benchmarks.manual_functions import BENCHMARK_FUNCTIONS

def test_quick_experiment():
    """Test experiment with a few functions."""
    
    print("=" * 60)
    print("🧪 QUICK TEST - CHECKING GLOBAL MINIMUMS")
    print("=" * 60)
    
    # Test with just a few functions that have global minimums
    test_functions = ['sphere', 'ackley', 'brent', 'booth', 'drop_wave']
    
    # Temporarily modify BENCHMARK_FUNCTIONS to only include test functions
    original_functions = BENCHMARK_FUNCTIONS.copy()
    
    # Filter to only test functions
    for func_name in list(BENCHMARK_FUNCTIONS.keys()):
        if func_name not in test_functions:
            del BENCHMARK_FUNCTIONS[func_name]
    
    try:
        # Run the experiment
        run_final_standard_experiment()
        print("\n✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during quick test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original functions
        BENCHMARK_FUNCTIONS.clear()
        BENCHMARK_FUNCTIONS.update(original_functions)

if __name__ == "__main__":
    test_quick_experiment()
