#!/usr/bin/env python3
"""
Web-based visualization system for benchmark experiment results
Creates interactive HTML pages with tables and charts
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import webbrowser
from datetime import datetime
import numpy as np

from experiment_runner import BenchmarkExperimentRunner
from benchmarks.functions import BenchmarkSuite

class WebVisualizer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.web_dir = self.results_dir / "web"
        self.web_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark = BenchmarkSuite()
        self.runner = BenchmarkExperimentRunner()
        
    def load_latest_results(self):
        """Load the most recent experiment results"""
        data_dir = self.results_dir / "data"
        
        # Find the most recent detailed results file
        json_files = list(data_dir.glob("detailed_results_*.json"))
        if not json_files:
            raise FileNotFoundError("No detailed results found. Run experiments first.")
        
        # Get the most recent file
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def create_comparison_dataframe(self, results, category_name):
        """Create DataFrame for comparison tables"""
        comparison_data = []
        
        for func_name in results:
            if func_name not in self.benchmark.functions:
                continue
                
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
                'GA_Best': np.min(ga_scores),
                'PSO_Best': np.min(pso_scores),
                'GA_Time': np.mean([run["run_time"] for run in ga_results]),
                'PSO_Time': np.mean([run["run_time"] for run in pso_results])
            })
        
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            # Calculate ranks
            df['GA_Rank'] = df['GA_Avg'].rank(method='min').astype(int)
            df['PSO_Rank'] = df['PSO_Avg'].rank(method='min').astype(int)
        
        return df
    
    def create_function_info_dataframe(self, functions):
        """Create DataFrame for function information"""
        info_data = []
        
        for func_name in functions:
            if func_name not in self.benchmark.functions:
                continue
                
            func_info = self.benchmark.get_function(func_name)
            
            # Get bounds and format as range
            bounds = func_info['bounds']
            if len(bounds) > 0:
                range_str = f"[{bounds[0][0]:.0f}, {bounds[0][1]:.0f}]"
            else:
                range_str = "N/A"
            
            info_data.append({
                'Function': func_name,
                'Range': range_str,
                'Dimension': func_info['dim'],
                'Min_Value': func_info['min_value'],
                'Type': 'Unimodal' if func_info['is_unimodal'] else 'Multimodal'
            })
        
        return pd.DataFrame(info_data)
    
    def create_comparison_table_html(self, df, title, filename):
        """Create interactive HTML table for comparison"""
        if df.empty:
            return
        
        # Format the data for display
        display_df = df.copy()
        display_df['GA_Avg'] = display_df['GA_Avg'].round(6)
        display_df['GA_Std'] = display_df['GA_Std'].round(6)
        display_df['PSO_Avg'] = display_df['PSO_Avg'].round(6)
        display_df['PSO_Std'] = display_df['PSO_Std'].round(6)
        display_df['GA_Time'] = display_df['GA_Time'].round(3)
        display_df['PSO_Time'] = display_df['PSO_Time'].round(3)
        
        # Create HTML table
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
            <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .table-container {{ margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin-bottom: 20px; }}
                .metric-card {{ background: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; }}
                .ga-color {{ color: #dc3545; font-weight: bold; }}
                .pso-color {{ color: #28a745; font-weight: bold; }}
                .rank-1 {{ background-color: #ffd700 !important; }}
                .rank-2 {{ background-color: #c0c0c0 !important; }}
                .rank-3 {{ background-color: #cd7f32 !important; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Performance comparison between Genetic Algorithm (GA) and Particle Swarm Optimization (PSO)</p>
            </div>
            
            <div class="container-fluid">
                <div class="row">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Total Functions</h5>
                            <h3>{len(df)}</h3>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>GA Wins</h5>
                            <h3 class="ga-color">{len(df[df['GA_Rank'] < df['PSO_Rank']])}</h3>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>PSO Wins</h5>
                            <h3 class="pso-color">{len(df[df['PSO_Rank'] < df['GA_Rank']])}</h3>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Ties</h5>
                            <h3>{len(df[df['GA_Rank'] == df['PSO_Rank']])}</h3>
                        </div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table id="comparisonTable" class="table table-striped table-hover">
                        <thead class="table-dark">
                            <tr>
                                <th>Function</th>
                                <th colspan="3" class="text-center ga-color">Genetic Algorithm (GA)</th>
                                <th colspan="3" class="text-center pso-color">Particle Swarm (PSO)</th>
                            </tr>
                            <tr>
                                <th></th>
                                <th>Average</th>
                                <th>Std Dev</th>
                                <th>Rank</th>
                                <th>Average</th>
                                <th>Std Dev</th>
                                <th>Rank</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for _, row in df.iterrows():
            rank_class_ga = f"rank-{row['GA_Rank']}" if row['GA_Rank'] <= 3 else ""
            rank_class_pso = f"rank-{row['PSO_Rank']}" if row['PSO_Rank'] <= 3 else ""
            
            html_content += f"""
                            <tr>
                                <td><strong>{row['Function']}</strong></td>
                                <td class="{rank_class_ga}">{row['GA_Avg']:.6f}</td>
                                <td>{row['GA_Std']:.6f}</td>
                                <td class="{rank_class_ga}"><strong>{row['GA_Rank']}</strong></td>
                                <td class="{rank_class_pso}">{row['PSO_Avg']:.6f}</td>
                                <td>{row['PSO_Std']:.6f}</td>
                                <td class="{rank_class_pso}"><strong>{row['PSO_Rank']}</strong></td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script>
                $(document).ready(function() {
                    $('#comparisonTable').DataTable({
                        "pageLength": 25,
                        "order": [[3, "asc"]], // Sort by GA rank
                        "columnDefs": [
                            { "orderable": false, "targets": 0 }
                        ]
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = self.web_dir / filename
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def create_function_info_html(self, df, title, filename):
        """Create HTML table for function information"""
        if df.empty:
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
            <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .table-container {{ margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin-bottom: 20px; }}
                .metric-card {{ background: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Function properties and characteristics</p>
            </div>
            
            <div class="container-fluid">
                <div class="row">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Total Functions</h5>
                            <h3>{len(df)}</h3>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Avg Dimension</h5>
                            <h3>{df['Dimension'].mean():.1f}</h3>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Min Dimension</h5>
                            <h3>{df['Dimension'].min()}</h3>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h5>Max Dimension</h5>
                            <h3>{df['Dimension'].max()}</h3>
                        </div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table id="functionTable" class="table table-striped table-hover">
                        <thead class="table-dark">
                            <tr>
                                <th>Function</th>
                                <th>Range</th>
                                <th>Dimension</th>
                                <th>Min Value</th>
                                <th>Type</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for _, row in df.iterrows():
            html_content += f"""
                            <tr>
                                <td><strong>{row['Function']}</strong></td>
                                <td>{row['Range']}</td>
                                <td>{row['Dimension']}</td>
                                <td>{row['Min_Value']:.6f}</td>
                                <td><span class="badge bg-{'primary' if row['Type'] == 'Unimodal' else 'secondary'}">{row['Type']}</span></td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script>
                $(document).ready(function() {
                    $('#functionTable').DataTable({
                        "pageLength": 25,
                        "order": [[0, "asc"]]
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = self.web_dir / filename
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def create_performance_charts(self, unimodal_df, multimodal_df):
        """Create performance comparison charts"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('GA vs PSO Performance (Unimodal)', 'GA vs PSO Performance (Multimodal)',
                          'Ranking Comparison (Unimodal)', 'Ranking Comparison (Multimodal)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance scatter plots
        if not unimodal_df.empty:
            fig.add_trace(
                go.Scatter(x=unimodal_df['GA_Avg'], y=unimodal_df['PSO_Avg'],
                          mode='markers+text', text=unimodal_df['Function'],
                          textposition="top center", name='Unimodal',
                          marker=dict(color='blue', size=8)),
                row=1, col=1
            )
        
        if not multimodal_df.empty:
            fig.add_trace(
                go.Scatter(x=multimodal_df['GA_Avg'], y=multimodal_df['PSO_Avg'],
                          mode='markers+text', text=multimodal_df['Function'],
                          textposition="top center", name='Multimodal',
                          marker=dict(color='red', size=8)),
                row=1, col=2
            )
        
        # Ranking comparison
        if not unimodal_df.empty:
            fig.add_trace(
                go.Bar(x=unimodal_df['Function'], y=unimodal_df['GA_Rank'],
                      name='GA Rank (Unimodal)', marker_color='lightblue'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=unimodal_df['Function'], y=unimodal_df['PSO_Rank'],
                      name='PSO Rank (Unimodal)', marker_color='lightgreen'),
                row=2, col=1
            )
        
        if not multimodal_df.empty:
            fig.add_trace(
                go.Bar(x=multimodal_df['Function'], y=multimodal_df['GA_Rank'],
                      name='GA Rank (Multimodal)', marker_color='lightcoral'),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=multimodal_df['Function'], y=multimodal_df['PSO_Rank'],
                      name='PSO Rank (Multimodal)', marker_color='lightyellow'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Benchmark Algorithm Performance Analysis",
            showlegend=True,
            height=800,
            template="plotly_white"
        )
        
        # Update axes
        fig.update_xaxes(title_text="GA Average Score", row=1, col=1)
        fig.update_yaxes(title_text="PSO Average Score", row=1, col=1)
        fig.update_xaxes(title_text="GA Average Score", row=1, col=2)
        fig.update_yaxes(title_text="PSO Average Score", row=1, col=2)
        fig.update_xaxes(title_text="Function", row=2, col=1)
        fig.update_yaxes(title_text="Rank", row=2, col=1)
        fig.update_xaxes(title_text="Function", row=2, col=2)
        fig.update_yaxes(title_text="Rank", row=2, col=2)
        
        # Save as HTML
        chart_file = self.web_dir / "performance_charts.html"
        fig.write_html(chart_file)
        
        return chart_file
    
    def create_dashboard_html(self, files):
        """Create main dashboard HTML page"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Optimization Results Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 0; text-align: center; }}
                .card {{ transition: transform 0.3s; }}
                .card:hover {{ transform: translateY(-5px); }}
                .card-body {{ text-align: center; }}
                .btn {{ margin: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Benchmark Optimization Results</h1>
                <p>Interactive analysis of Genetic Algorithm vs Particle Swarm Optimization performance</p>
                <p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
            
            <div class="container mt-5">
                <div class="row">
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title">📊 Unimodal Functions Comparison</h5>
                                <p class="card-text">GA vs PSO performance comparison for unimodal benchmark functions</p>
                                <a href="{files['unimodal_comparison'].name}" class="btn btn-primary" target="_blank">View Table</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title">📊 Multimodal Functions Comparison</h5>
                                <p class="card-text">GA vs PSO performance comparison for multimodal benchmark functions</p>
                                <a href="{files['multimodal_comparison'].name}" class="btn btn-primary" target="_blank">View Table</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title">📋 Unimodal Function Information</h5>
                                <p class="card-text">Properties and characteristics of unimodal benchmark functions</p>
                                <a href="{files['unimodal_info'].name}" class="btn btn-info" target="_blank">View Table</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            <div class="card-body">
                                <h5 class="card-title">📋 Multimodal Function Information</h5>
                                <p class="card-text">Properties and characteristics of multimodal benchmark functions</p>
                                <a href="{files['multimodal_info'].name}" class="btn btn-info" target="_blank">View Table</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-12 mb-4">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">📈 Performance Charts & Graphs</h5>
                                <p class="card-text">Interactive visualizations comparing algorithm performance</p>
                                <a href="{files['charts'].name}" class="btn btn-success" target="_blank">View Charts</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        dashboard_file = self.web_dir / "dashboard.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return dashboard_file
    
    def generate_all_visualizations(self):
        """Generate all web visualizations"""
        
        print("🌐 Loading experiment results...")
        results = self.load_latest_results()
        
        print("📊 Creating comparison dataframes...")
        unimodal_comparison = self.create_comparison_dataframe(
            results['unimodal_results']['results'], 'Unimodal')
        multimodal_comparison = self.create_comparison_dataframe(
            results['multimodal_results']['results'], 'Multimodal')
        
        print("📋 Creating function info dataframes...")
        unimodal_info = self.create_function_info_dataframe(self.runner.unimodal_functions)
        multimodal_info = self.create_function_info_dataframe(self.runner.multimodal_functions)
        
        print("🎨 Generating HTML tables...")
        files = {}
        files['unimodal_comparison'] = self.create_comparison_table_html(
            unimodal_comparison, "Unimodal Functions - GA vs PSO Comparison", "unimodal_comparison.html")
        files['multimodal_comparison'] = self.create_comparison_table_html(
            multimodal_comparison, "Multimodal Functions - GA vs PSO Comparison", "multimodal_comparison.html")
        files['unimodal_info'] = self.create_function_info_html(
            unimodal_info, "Unimodal Functions - Function Information", "unimodal_function_info.html")
        files['multimodal_info'] = self.create_function_info_html(
            multimodal_info, "Multimodal Functions - Function Information", "multimodal_function_info.html")
        
        print("📈 Creating performance charts...")
        files['charts'] = self.create_performance_charts(unimodal_comparison, multimodal_comparison)
        
        print("🏠 Creating dashboard...")
        dashboard_file = self.create_dashboard_html(files)
        
        print("\n✅ All web visualizations generated successfully!")
        print("📁 Files saved in 'results/web/':")
        for name, file_path in files.items():
            print(f"   - {name}: {file_path.name}")
        print(f"   - dashboard: {dashboard_file.name}")
        
        return dashboard_file, files
    
    def open_dashboard(self):
        """Open the dashboard in the default web browser"""
        dashboard_file, _ = self.generate_all_visualizations()
        
        print(f"\n🌐 Opening dashboard in browser...")
        webbrowser.open(f"file://{dashboard_file.absolute()}")
        
        return dashboard_file

if __name__ == "__main__":
    visualizer = WebVisualizer()
    
    print("🎯 Web Visualization System")
    print("=" * 40)
    
    # Generate and open dashboard
    dashboard_file = visualizer.open_dashboard()
    
    print(f"\n🎉 Dashboard opened: {dashboard_file}")
    print("📁 All visualization files are in 'results/web/' folder")

