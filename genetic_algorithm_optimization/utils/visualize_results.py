"""
Visualization tools for optimization results.

This module provides plotting capabilities to visualize algorithm performance,
convergence curves, and statistical comparisons.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import os


def plot_convergence_curves(results_dir="results", save_plots=True):
    """
    Plot convergence curves for different algorithms.
    
    Args:
        results_dir: Directory containing results
        save_plots: Whether to save plots to files
    """
    plt.style.use('seaborn-v0_8')
    
    # Find convergence data files
    convergence_files = [f for f in os.listdir(results_dir) 
                       if 'convergence' in f and f.endswith('.csv')]
    
    if not convergence_files:
        print("No convergence data found.")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, file in enumerate(convergence_files[:4]):  # Plot first 4 files
        if i >= 4:
            break
            
        filepath = os.path.join(results_dir, file)
        df = pd.read_csv(filepath)
        
        ax = axes[i]
        
        # Plot convergence for each run
        for run in df['run'].unique():
            run_data = df[df['run'] == run]
            ax.plot(run_data['evaluation'], run_data['best_fitness'], 
                   alpha=0.3, linewidth=0.5)
        
        # Plot mean convergence
        mean_convergence = df.groupby('evaluation')['best_fitness'].mean()
        ax.plot(mean_convergence.index, mean_convergence.values, 
               linewidth=2, color='red', label='Mean')
        
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Best Fitness')
        ax.set_title(f'Convergence: {file.replace("_convergence_", " - ").replace(".csv", "")}')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(results_dir, 'convergence_curves.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Convergence curves saved to {results_dir}/convergence_curves.png")
    
    plt.show()


def plot_algorithm_comparison(results_dir="results", save_plots=True):
    """
    Plot algorithm comparison charts.
    
    Args:
        results_dir: Directory containing results
        save_plots: Whether to save plots to files
    """
    plt.style.use('seaborn-v0_8')
    
    # Find comparison files
    comparison_files = [f for f in os.listdir(results_dir) 
                       if 'comparison' in f and f.endswith('.csv')]
    
    if not comparison_files:
        print("No comparison data found.")
        return
    
    # Load the most recent comparison file
    latest_file = max(comparison_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    filepath = os.path.join(results_dir, latest_file)
    df = pd.read_csv(filepath)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean fitness comparison
    ax1 = axes[0, 0]
    pivot_mean = df.pivot_table(values='mean_fitness', index='function', columns='algorithm', aggfunc='mean')
    pivot_mean.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Mean Fitness Comparison')
    ax1.set_ylabel('Mean Fitness')
    ax1.set_xlabel('Function')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_yscale('log')
    
    # 2. Success rate comparison
    ax2 = axes[0, 1]
    if 'success_rate' in df.columns:
        pivot_success = df.pivot_table(values='success_rate', index='function', columns='algorithm', aggfunc='mean')
        pivot_success.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Success Rate Comparison')
        ax2.set_ylabel('Success Rate')
        ax2.set_xlabel('Function')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax2.text(0.5, 0.5, 'Success rate data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Success Rate Comparison (No Data)')
    
    # 3. Box plot of fitness values
    ax3 = axes[1, 0]
    df_melted = df.melt(id_vars=['function', 'algorithm'], 
                       value_vars=['mean_fitness'], 
                       var_name='metric', value_name='value')
    sns.boxplot(data=df_melted, x='function', y='value', hue='algorithm', ax=ax3)
    ax3.set_title('Fitness Distribution by Function and Algorithm')
    ax3.set_ylabel('Mean Fitness')
    ax3.set_xlabel('Function')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yscale('log')
    
    # 4. Algorithm performance heatmap
    ax4 = axes[1, 1]
    pivot_heatmap = df.pivot_table(values='mean_fitness', index='function', columns='algorithm', aggfunc='mean')
    sns.heatmap(pivot_heatmap, annot=True, fmt='.2e', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Algorithm Performance Heatmap')
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Function')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(results_dir, 'algorithm_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Algorithm comparison saved to {results_dir}/algorithm_comparison.png")
    
    plt.show()


def plot_parallel_vs_sequential(results_dir="results_parallel", save_plots=True):
    """
    Plot parallel vs sequential performance comparison.
    
    Args:
        results_dir: Directory containing parallel results
        save_plots: Whether to save plots to files
    """
    plt.style.use('seaborn-v0_8')
    
    # Find parallel comparison files
    comparison_files = [f for f in os.listdir(results_dir) 
                       if 'comparison' in f and f.endswith('.csv')]
    
    if not comparison_files:
        print("No parallel comparison data found.")
        return
    
    # Load the most recent comparison file
    latest_file = max(comparison_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    filepath = os.path.join(results_dir, latest_file)
    df = pd.read_csv(filepath)
    
    # Separate parallel and sequential algorithms
    parallel_algs = df[df['algorithm'].str.contains('Parallel', na=False)]
    sequential_algs = df[~df['algorithm'].str.contains('Parallel', na=False)]
    
    if parallel_algs.empty or sequential_algs.empty:
        print("Insufficient data for parallel vs sequential comparison.")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance comparison
    ax1 = axes[0, 0]
    comparison_data = []
    
    for function in df['function'].unique():
        func_data = df[df['function'] == function]
        for alg in func_data['algorithm'].unique():
            alg_data = func_data[func_data['algorithm'] == alg]
            comparison_data.append({
                'function': function,
                'algorithm': alg,
                'mean_fitness': alg_data['mean_fitness'].mean(),
                'is_parallel': 'Parallel' in alg
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Group by function and algorithm type
    pivot_comparison = comparison_df.pivot_table(
        values='mean_fitness', 
        index='function', 
        columns='is_parallel', 
        aggfunc='mean'
    )
    
    pivot_comparison.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Parallel vs Sequential Performance')
    ax1.set_ylabel('Mean Fitness')
    ax1.set_xlabel('Function')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(['Sequential', 'Parallel'])
    ax1.set_yscale('log')
    
    # 2. Speed improvement
    ax2 = axes[0, 1]
    speed_improvement = []
    
    for function in df['function'].unique():
        func_data = df[df['function'] == function]
        parallel_data = func_data[func_data['algorithm'].str.contains('Parallel', na=False)]
        sequential_data = func_data[~func_data['algorithm'].str.contains('Parallel', na=False)]
        
        if not parallel_data.empty and not sequential_data.empty:
            parallel_mean = parallel_data['mean_fitness'].mean()
            sequential_mean = sequential_data['mean_fitness'].mean()
            improvement = (sequential_mean - parallel_mean) / sequential_mean * 100
            speed_improvement.append({
                'function': function,
                'improvement': improvement
            })
    
    if speed_improvement:
        improvement_df = pd.DataFrame(speed_improvement)
        improvement_df.plot(x='function', y='improvement', kind='bar', ax=ax2, color='green')
        ax2.set_title('Performance Improvement with Parallel Processing')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_xlabel('Function')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 3. Algorithm efficiency comparison
    ax3 = axes[1, 0]
    efficiency_data = comparison_df.groupby(['function', 'is_parallel'])['mean_fitness'].mean().unstack()
    efficiency_data.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Algorithm Efficiency Comparison')
    ax3.set_ylabel('Mean Fitness')
    ax3.set_xlabel('Function')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(['Sequential', 'Parallel'])
    ax3.set_yscale('log')
    
    # 4. Success rate comparison
    ax4 = axes[1, 1]
    if 'success_rate' in df.columns:
        success_data = comparison_df.groupby(['function', 'is_parallel'])['mean_fitness'].mean().unstack()
        success_data.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Success Rate Comparison')
        ax4.set_ylabel('Success Rate')
        ax4.set_xlabel('Function')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(['Sequential', 'Parallel'])
    else:
        ax4.text(0.5, 0.5, 'Success rate data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Success Rate Comparison (No Data)')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(results_dir, 'parallel_vs_sequential.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Parallel vs sequential comparison saved to {results_dir}/parallel_vs_sequential.png")
    
    plt.show()


def create_summary_dashboard(results_dir="results", save_plots=True):
    """
    Create a comprehensive summary dashboard.
    
    Args:
        results_dir: Directory containing results
        save_plots: Whether to save plots to files
    """
    print("Creating summary dashboard...")
    
    # Create all plots
    plot_convergence_curves(results_dir, save_plots)
    plot_algorithm_comparison(results_dir, save_plots)
    
    # Check if parallel results exist
    parallel_dir = "results_parallel"
    if os.path.exists(parallel_dir):
        plot_parallel_vs_sequential(parallel_dir, save_plots)
    
    print("Summary dashboard created!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "parallel":
        create_summary_dashboard("results_parallel")
    else:
        create_summary_dashboard("results")
