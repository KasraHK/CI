import numpy as np
import benchmarkfcns

class BenchmarkSuite:
    def __init__(self):
        self.functions = {}
        self._setup_functions()
    
    def _setup_functions(self):
        # Get all available function names from benchmarkfcns
        all_function_names = [
            name for name in dir(benchmarkfcns) 
            if not name.startswith('_') and callable(getattr(benchmarkfcns, name))
        ]
        
        # Register all functions
        for fname in all_function_names:
            try:
                func = getattr(benchmarkfcns, fname)
                
                # Get function properties
                bounds = getattr(func, 'bounds', [[-100, 100]])
                dim = getattr(func, 'dim', 30)
                min_value = getattr(func, 'min_value', 0.0)
                
                # Determine if function is unimodal (you'll need to categorize these)
                is_unimodal = fname in [
                    'sphere', 'ellipsoid', 'sum_squares', 'rotated_ellipsoid', 
                    'discus', 'bent_cigar', 'different_powers', 'schwefel_12',
                    'schwefel_221', 'schwefel_222', 'rosenbrock', 'dixon_price',
                    'rotated_rosenbrock', 'rotated_schwefel_12', 'rotated_schwefel_221',
                    'rotated_schwefel_222', 'rotated_discus', 'different_powers_rotated',
                    'bent_cigar_rotated', 'rotated_bent_cigar', 'rotated_different_powers'
                ]
                
                self.functions[fname] = {
                    'func': func,
                    'bounds': np.array(bounds),
                    'dim': dim,
                    'min_value': min_value,
                    'is_unimodal': is_unimodal
                }
                
            except Exception as e:
                print(f"Warning: Could not load function {fname}: {e}")
    
    def get_function(self, name):
        if name not in self.functions:
            raise ValueError(f"Function {name} not available")
        return self.functions[name]
    
    def evaluate(self, name, x):
        func_info = self.get_function(name)
        return func_info['func'](x)
    
    def get_bounds(self, name, dim=None):
        func_info = self.get_function(name)
        if dim is not None:
            # Return bounds for specified dimension
            return np.tile(func_info['bounds'][0], (dim, 1))
        return np.tile(func_info['bounds'][0], (func_info['dim'], 1))
    
    def get_dimension(self, name):
        return self.get_function(name)['dim']
    
    def get_min_value(self, name):
        return self.get_function(name)['min_value']
    
    def is_unimodal(self, name):
        return self.get_function(name)['is_unimodal']
    
    def get_all_functions(self):
        return list(self.functions.keys())
    
    def get_unimodal_functions(self):
        return [f for f in self.functions if self.functions[f]['is_unimodal']]
    
    def get_multimodal_functions(self):
        return [f for f in self.functions if not self.functions[f]['is_unimodal']]
    
    def get_function_info(self, name):
        """Get complete information about a function for table generation"""
        func_info = self.get_function(name)
        return {
            'name': name,
            'range': f"[{func_info['bounds'][0][0]}, {func_info['bounds'][0][1]}]",
            'dim': func_info['dim'],
            'fmin': func_info['min_value'],
            'type': 'unimodal' if func_info['is_unimodal'] else 'multimodal'
        }
    
    # def get_all_function_names():
    #     benchmark = BenchmarkSuite()
    #     all_funcs = benchmark.get_all_functions()
        
    #     # Blacklist problematic functions
    #     blacklist = [
    #         'ackleyn2', 'ackleyn3', 'ackleyn4',  # Functions that require specific input formats
    #         # Add other problematic functions here
    #     ]
        
    #     return [f for f in all_funcs if f not in blacklist]