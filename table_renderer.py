#!/usr/bin/env python3
"""
Table renderer for benchmark experiment results
Creates formatted tables for unimodal and multimodal functions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from experiment_runner import BenchmarkExperimentRunner
from benchmarks.functions import BenchmarkSuite
from config import F_MIN_INFO

class TableRenderer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.tables_dir = self.results_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark = BenchmarkSuite()
        self.runner = BenchmarkExperimentRunner()
        
    def load_latest_results(self):
        """Load the most recent experiment results
        
        Returns:
            dict: Dictionary containing the most recent experiment results
            
        Raises:
            FileNotFoundError: If no detailed results files are found
        """
        data_dir = self.results_dir / "data"
        
        # Find the most recent detailed results file
        json_files = list(data_dir.glob("detailed_results_*.json"))
        if not json_files:
            raise FileNotFoundError("No detailed results found. Run experiments first.")
        
        # Get the most recent file
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        import json
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def create_comparison_table(self, results, category_name):
        """Create GA vs PSO comparison table with avg, std, rank
        
        Args:
            results (dict): Dictionary containing experiment results for each function
            category_name (str): Name of the category (e.g., "Unimodal", "Multimodal")
            
        Returns:
            pandas.DataFrame: Comparison table with statistics and rankings
        """
        
        # Prepare data for comparison
        comparison_data = []
        
        for func_name in results:
            if func_name not in self.benchmark.functions:
                continue
                
            func_info = self.benchmark.get_function(func_name)
            
            # Get results for both algorithms
            ga_results = results[func_name].get('ga', [])
            pso_results = results[func_name].get('pso', [])
            
            if not ga_results or not pso_results:
                continue
            
            # Calculate statistics
            ga_scores = [run["best_score"] for run in ga_results]
            pso_scores = [run["best_score"] for run in pso_results]
            
            comparison_data.append({
                'Function': func_name,
                'GA_Avg': np.mean(ga_scores),
                'GA_Std': np.std(ga_scores),
                'PSO_Avg': np.mean(pso_scores),
                'PSO_Std': np.std(pso_scores),
                'GA_Scores': ga_scores,
                'PSO_Scores': pso_scores
            })
        
        # Calculate ranks
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            # Rank by average score (lower is better)
            df['GA_Rank'] = df['GA_Avg'].rank(method='min').astype(int)
            df['PSO_Rank'] = df['PSO_Avg'].rank(method='min').astype(int)
        
        return df
    
    def create_function_info_table(self, functions, category_name):
        """Create function information table with range, dim, fmin
        
        Args:
            functions (list): List of function names
            category_name (str): Name of the category (e.g., "Unimodal", "Multimodal")
            
        Returns:
            pandas.DataFrame: Function information table
        """
        
        info_data = []
        
        for func_name in functions:
            if func_name not in self.benchmark.functions:
                continue
                
            func_info = self.benchmark.get_function(func_name)
            
            # Get reasonable bounds and format as range
            bounds = self.benchmark.get_reasonable_bounds(func_name, func_info['dim'])
            if len(bounds) > 0:
                range_str = f"[{bounds[0][0]:.0f}, {bounds[0][1]:.0f}]"
            else:
                range_str = "N/A"
            
            # Get Fmin value from F_MIN_INFO, with fallback to function's min_value
            fmin_value = self._get_fmin_value(func_name)
            
            info_data.append({
                'Function': func_name,
                'Range': range_str,
                'Dim': func_info['dim'],
                'Fmin': fmin_value
            })
        
        return pd.DataFrame(info_data)
    
    def _get_fmin_value(self, func_name):
        """Get the minimum value for a function from F_MIN_INFO
        
        Args:
            func_name (str): Function name from benchmark
            
        Returns:
            float: Minimum value for the function
        """
        # Create mapping from benchmark function names to F_MIN_INFO keys
        name_mapping = {
            'ackleyn2': 'ackley_n2',
            'ackleyn3': 'ackley_n3', 
            'ackleyn4': 'ackley_n4',
            'alpine1': 'alpine_n1',
            'alpine2': 'alpine_n2',
            'bartelsconn': 'bartels_conn',
            'bohachevskyn1': 'bohachevsky_n1',
            'bohachevskyn2': 'bohachevsky_n2',
            'bukinn6': 'bukin_n6',
            'carromtable': 'carrom_table',
            'crossintray': 'cross_in_tray',
            'deckersaarts': 'deckers_aarts',
            'eggcrate': 'egg_crate',
            'elattarvidyasagardutta': 'el_attar_vidyasagar_dutta',
            'goldsteinprice': 'goldstein_price',
            'gramacylee': 'gramacy_lee',
            'happycat': 'happy_cat',
            'holdertable': 'holder_table',
            'levin13': 'levi_n13',
            'powellsum': 'powell_sum',
            'schaffern1': 'schaffer_n1',
            'schaffern2': 'schaffer_n2',
            'schaffern3': 'schaffer_n3',
            'schaffern4': 'schaffer_n4',
            'schwefel220': 'schwefel_2_20',
            'schwefel221': 'schwefel_2_21',
            'schwefel222': 'schwefel_2_22',
            'schwefel223': 'schwefel_2_23',
            'shubert3': 'shubert_3',
            'shubertn4': 'shubert_n4',
            'sumsquares': 'sum_squares',
            'styblinskitank': 'styblinski_tang',
            'threehumpcamel': 'three_hump_camel',
            'xinsheyang': 'xin_she_yang',
            'xinsheyangn2': 'xin_she_yang_n2',
            'xinsheyangn3': 'xin_she_yang_n3',
            'xinsheyangn4': 'xin_she_yang_n4'
        }
        
        # Get the mapped name or use original name
        mapped_name = name_mapping.get(func_name, func_name)
        
        # Look up in F_MIN_INFO
        if mapped_name in F_MIN_INFO:
            return F_MIN_INFO[mapped_name][0]  # Return the first element (min value)
        else:
            # Fallback to function's min_value if not found in F_MIN_INFO
            func_info = self.benchmark.get_function(func_name)
            return func_info['min_value']
    
    def format_comparison_table(self, df):
        """Format the comparison table for display
        
        Args:
            df (pandas.DataFrame): Comparison table to format
            
        Returns:
            str: Formatted table string for console display
        """
        if df.empty:
            return "No data available"
        
        # Create formatted table
        lines = []
        lines.append("_" * 80)
        lines.append(f"{'Function':<15} {'GA':<20} {'PSO':<20}")
        lines.append(f"{'':<15} {'Avg':<8} {'Std':<8} {'Rank':<4} {'Avg':<8} {'Std':<8} {'Rank':<4}")
        lines.append("_" * 80)
        
        for _, row in df.iterrows():
            func_name = row['Function']
            ga_avg = f"{row['GA_Avg']:.6f}"
            ga_std = f"{row['GA_Std']:.6f}"
            ga_rank = f"{row['GA_Rank']}"
            pso_avg = f"{row['PSO_Avg']:.6f}"
            pso_std = f"{row['PSO_Std']:.6f}"
            pso_rank = f"{row['PSO_Rank']}"
            
            lines.append(f"{func_name:<15} {ga_avg:<8} {ga_std:<8} {ga_rank:<4} {pso_avg:<8} {pso_std:<8} {pso_rank:<4}")
        
        lines.append("_" * 80)
        return "\n".join(lines)
    
    def format_function_info_table(self, df):
        """Format the function information table for display
        
        Args:
            df (pandas.DataFrame): Function info table to format
            
        Returns:
            str: Formatted table string for console display
        """
        """Format the function info table for display"""
        if df.empty:
            return "No data available"
        
        lines = []
        lines.append("_" * 60)
        lines.append(f"{'Function':<15} {'Range':<15} {'Dim':<6} {'Fmin':<10}")
        lines.append("_" * 60)
        
        for _, row in df.iterrows():
            func_name = row['Function']
            range_str = row['Range']
            dim = f"{row['Dim']}"
            fmin = f"{row['Fmin']:.6f}"
            
            lines.append(f"{func_name:<15} {range_str:<15} {dim:<6} {fmin:<10}")
        
        lines.append("_" * 60)
        return "\n".join(lines)
    
    def save_tables(self, unimodal_comparison, multimodal_comparison, 
                   unimodal_info, multimodal_info):
        """Save all tables to files"""
        
        # Save comparison tables
        unimodal_comp_file = self.tables_dir / "unimodal_comparison.txt"
        multimodal_comp_file = self.tables_dir / "multimodal_comparison.txt"
        
        with open(unimodal_comp_file, 'w') as f:
            f.write("UNIMODAL FUNCTIONS - GA vs PSO COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(self.format_comparison_table(unimodal_comparison))
        
        with open(multimodal_comp_file, 'w') as f:
            f.write("MULTIMODAL FUNCTIONS - GA vs PSO COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(self.format_comparison_table(multimodal_comparison))
        
        # Save function info tables
        unimodal_info_file = self.tables_dir / "unimodal_function_info.txt"
        multimodal_info_file = self.tables_dir / "multimodal_function_info.txt"
        
        with open(unimodal_info_file, 'w') as f:
            f.write("UNIMODAL FUNCTIONS - FUNCTION INFORMATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(self.format_function_info_table(unimodal_info))
        
        with open(multimodal_info_file, 'w') as f:
            f.write("MULTIMODAL FUNCTIONS - FUNCTION INFORMATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(self.format_function_info_table(multimodal_info))
        
        return {
            'unimodal_comparison': unimodal_comp_file,
            'multimodal_comparison': multimodal_comp_file,
            'unimodal_info': unimodal_info_file,
            'multimodal_info': multimodal_info_file
        }
    
    def generate_all_tables(self):
        """Generate and save all formatted tables
        
        Returns:
            list: List of file paths for generated table files
        """
        """Generate all 4 tables from the latest experiment results"""
        
        print("📊 Loading experiment results...")
        results = self.load_latest_results()
        
        print("🔧 Creating comparison tables...")
        unimodal_comparison = self.create_comparison_table(
            results['unimodal_results']['results'], 'Unimodal')
        multimodal_comparison = self.create_comparison_table(
            results['multimodal_results']['results'], 'Multimodal')
        
        print("📋 Creating function info tables...")
        unimodal_info = self.create_function_info_table(
            self.runner.unimodal_functions, 'Unimodal')
        multimodal_info = self.create_function_info_table(
            self.runner.multimodal_functions, 'Multimodal')
        
        print("💾 Saving tables...")
        file_paths = self.save_tables(
            unimodal_comparison, multimodal_comparison,
            unimodal_info, multimodal_info
        )
        
        print("\n✅ All tables generated successfully!")
        print("📁 Files saved in 'results/tables/':")
        for name, path in file_paths.items():
            print(f"   - {name}: {path.name}")
        
        return file_paths
    
    def display_tables(self):
        """Display all tables in the console"""
        
        print("📊 Loading experiment results...")
        results = self.load_latest_results()
        
        print("🔧 Creating comparison tables...")
        unimodal_comparison = self.create_comparison_table(
            results['unimodal_results']['results'], 'Unimodal')
        multimodal_comparison = self.create_comparison_table(
            results['multimodal_results']['results'], 'Multimodal')
        
        print("📋 Creating function info tables...")
        unimodal_info = self.create_function_info_table(
            self.runner.unimodal_functions, 'Unimodal')
        multimodal_info = self.create_function_info_table(
            self.runner.multimodal_functions, 'Multimodal')
        
        print("\n" + "="*80)
        print("UNIMODAL FUNCTIONS - GA vs PSO COMPARISON")
        print("="*80)
        print(self.format_comparison_table(unimodal_comparison))
        
        print("\n" + "="*80)
        print("MULTIMODAL FUNCTIONS - GA vs PSO COMPARISON")
        print("="*80)
        print(self.format_comparison_table(multimodal_comparison))
        
        print("\n" + "="*60)
        print("UNIMODAL FUNCTIONS - FUNCTION INFORMATION")
        print("="*60)
        print(self.format_function_info_table(unimodal_info))
        
        print("\n" + "="*60)
        print("MULTIMODAL FUNCTIONS - FUNCTION INFORMATION")
        print("="*60)
        print(self.format_function_info_table(multimodal_info))

if __name__ == "__main__":
    renderer = TableRenderer()
    
    print("🎯 Benchmark Table Renderer")
    print("=" * 40)
    
    # Generate and display tables
    renderer.display_tables()
    
    # Also save to files
    print("\n💾 Saving tables to files...")
    renderer.generate_all_tables()

