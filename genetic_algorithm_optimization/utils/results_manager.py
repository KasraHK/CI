"""
Results management system for optimization experiments.

This module handles saving, loading, and analyzing experimental results
in CSV format for easy data analysis and presentation.
"""

import csv
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


class ResultsManager:
    """Manages experimental results and saves them to CSV files."""
    
    def __init__(self, results_dir="results"):
        """
        Initialize the results manager.
        
        Args:
            results_dir: Directory to save results files
        """
        self.results_dir = results_dir
        self.ensure_results_dir()
    
    def ensure_results_dir(self):
        """Create results directory if it doesn't exist."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def save_experiment_results(self, experiment_name, algorithm_name, function_name, 
                               results, metadata=None):
        """
        Save experiment results to CSV file.
        
        Args:
            experiment_name: Name of the experiment
            algorithm_name: Name of the algorithm used
            function_name: Name of the benchmark function
            results: List of dictionaries containing results
            metadata: Additional metadata to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{algorithm_name}_{function_name}_{timestamp}.csv"
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare data for CSV
        csv_data = []
        for i, result in enumerate(results):
            row = {
                'run': i + 1,
                'algorithm': algorithm_name,
                'function': function_name,
                'best_fitness': result['best_fitness'],
                'mean_fitness': result['mean_fitness'],
                'std_fitness': result['std_fitness'],
                'worst_fitness': result['worst_fitness'],
                'evaluations': result['evaluations'],
                'generations': result.get('generations', 'N/A'),
                'convergence_time': result.get('convergence_time', 'N/A'),
                'timestamp': timestamp
            }
            
            # Add algorithm-specific parameters
            if 'parameters' in result:
                for key, value in result['parameters'].items():
                    row[f'param_{key}'] = value
            
            # Add best solution (first 5 dimensions)
            if 'best_solution' in result:
                solution = result['best_solution']
                if hasattr(solution, 'genes'):
                    solution = solution.genes
                for j in range(min(5, len(solution))):
                    row[f'best_dim_{j+1}'] = solution[j]
            
            csv_data.append(row)
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        # Save metadata as JSON
        if metadata:
            metadata_file = filepath.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def save_comparison_results(self, experiment_name, comparison_results):
        """
        Save comparison results between different algorithms/methods.
        
        Args:
            experiment_name: Name of the experiment
            comparison_results: Dictionary with comparison data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_comparison_{timestamp}.csv"
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare comparison data
        csv_data = []
        for function_name, function_results in comparison_results.items():
            for algorithm_name, algorithm_results in function_results.items():
                row = {
                    'function': function_name,
                    'algorithm': algorithm_name,
                    'mean_fitness': algorithm_results['mean_fitness'],
                    'std_fitness': algorithm_results['std_fitness'],
                    'best_fitness': algorithm_results['best_fitness'],
                    'worst_fitness': algorithm_results['worst_fitness'],
                    'success_rate': algorithm_results.get('success_rate', 'N/A'),
                    'convergence_speed': algorithm_results.get('convergence_speed', 'N/A'),
                    'timestamp': timestamp
                }
                
                # Add algorithm parameters
                if 'parameters' in algorithm_results:
                    for key, value in algorithm_results['parameters'].items():
                        row[f'param_{key}'] = value
                
                csv_data.append(row)
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"Comparison results saved to: {filepath}")
        return filepath
    
    def save_convergence_data(self, experiment_name, algorithm_name, function_name, 
                             convergence_data):
        """
        Save convergence data (fitness over generations).
        
        Args:
            experiment_name: Name of the experiment
            algorithm_name: Name of the algorithm
            function_name: Name of the function
            convergence_data: List of convergence records
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_convergence_{algorithm_name}_{function_name}_{timestamp}.csv"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['generation', 'evaluation', 'best_fitness', 'mean_fitness', 
                         'worst_fitness', 'diversity', 'run']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(convergence_data)
        
        print(f"Convergence data saved to: {filepath}")
        return filepath
    
    def load_results(self, filepath):
        """Load results from CSV file."""
        return pd.read_csv(filepath)
    
    def generate_summary_report(self, experiment_name):
        """Generate a summary report of all experiments."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"{experiment_name}_summary_{timestamp}.txt")
        
        # Find all CSV files for this experiment
        csv_files = [f for f in os.listdir(self.results_dir) 
                    if f.startswith(experiment_name) and f.endswith('.csv')]
        
        with open(report_file, 'w') as f:
            f.write(f"EXPERIMENT SUMMARY REPORT\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"Found {len(csv_files)} result files:\n")
            for csv_file in csv_files:
                f.write(f"  - {csv_file}\n")
            
            f.write(f"\nDetailed Analysis:\n")
            f.write(f"{'-'*30}\n")
            
            for csv_file in csv_files:
                filepath = os.path.join(self.results_dir, csv_file)
                try:
                    df = pd.read_csv(filepath)
                    f.write(f"\nFile: {csv_file}\n")
                    f.write(f"Records: {len(df)}\n")
                    
                    if 'best_fitness' in df.columns:
                        f.write(f"Best Fitness - Mean: {df['best_fitness'].mean():.6f}, "
                               f"Std: {df['best_fitness'].std():.6f}\n")
                        f.write(f"Best Fitness - Min: {df['best_fitness'].min():.6f}, "
                               f"Max: {df['best_fitness'].max():.6f}\n")
                    
                    if 'evaluations' in df.columns:
                        f.write(f"Evaluations - Mean: {df['evaluations'].mean():.0f}\n")
                    
                except Exception as e:
                    f.write(f"Error reading {csv_file}: {str(e)}\n")
        
        print(f"Summary report saved to: {report_file}")
        return report_file
    
    def get_best_performing_algorithm(self, function_name):
        """Get the best performing algorithm for a specific function."""
        csv_files = [f for f in os.listdir(self.results_dir) 
                    if function_name in f and f.endswith('.csv')]
        
        best_algorithm = None
        best_fitness = float('inf')
        
        for csv_file in csv_files:
            filepath = os.path.join(self.results_dir, csv_file)
            try:
                df = pd.read_csv(filepath)
                if 'best_fitness' in df.columns:
                    mean_fitness = df['best_fitness'].mean()
                    if mean_fitness < best_fitness:
                        best_fitness = mean_fitness
                        best_algorithm = df['algorithm'].iloc[0]
            except Exception as e:
                print(f"Error reading {csv_file}: {str(e)}")
        
        return best_algorithm, best_fitness


def create_results_summary_table(results_dir="results"):
    """Create a summary table of all results."""
    manager = ResultsManager(results_dir)
    
    # Find all comparison CSV files
    csv_files = [f for f in os.listdir(results_dir) 
                if 'comparison' in f and f.endswith('.csv')]
    
    if not csv_files:
        print("No comparison files found.")
        return None
    
    # Load the most recent comparison file
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    filepath = os.path.join(results_dir, latest_file)
    
    df = pd.read_csv(filepath)
    
    # Create pivot table
    pivot_table = df.pivot_table(
        values='mean_fitness', 
        index='function', 
        columns='algorithm', 
        aggfunc='mean'
    )
    
    print("SUMMARY TABLE - Mean Fitness by Function and Algorithm:")
    print("="*60)
    print(pivot_table.round(6))
    
    return pivot_table
