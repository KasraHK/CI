#!/usr/bin/env python3
"""
Main entry point for the Benchmark Optimization Project
"""

from experiment_runner import BenchmarkExperimentRunner
from table_renderer import TableRenderer
from simple_web_visualizer import SimpleWebVisualizer
from config import EXPERIMENT_CONFIG

def main():
    
    # # Initialize the experiment runner
    runner = BenchmarkExperimentRunner(results_dir=EXPERIMENT_CONFIG["results_directory"])

    
    # unimodal_table, multimodal_table = runner.run_all_experiments().
    print(runner.run_single_experiment('elattar', 'ga', runs=5))

    
    # # Generate formatted tables
    # renderer = TableRenderer(results_dir=EXPERIMENT_CONFIG["results_directory"])
    # renderer.generate_all_tables()
    
    # # Generate web visualizations
    # web_visualizer = SimpleWebVisualizer(results_dir=EXPERIMENT_CONFIG["results_directory"])
    # dashboard_file = web_visualizer.open_dashboard()

if __name__ == "__main__":
    main()

