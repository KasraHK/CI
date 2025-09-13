import numpy as np
import benchmarkfcns

class BenchmarkSuite:
    def __init__(self):
        self.functions = {}
        self._setup_functions()
    
    def _setup_functions(self):
        """Setup and register all available benchmark functions
        
        This method discovers all available functions from the benchmarkfcns library,
        categorizes them as unimodal or multimodal, and stores their properties.
        """
        # Get all available function names from benchmarkfcns
        all_function_names = [
            name for name in dir(benchmarkfcns) 
            if not name.startswith('_') and callable(getattr(benchmarkfcns, name))
        ]
        # print(all_function_names)
        
        # Register all functions
        for fname in all_function_names:
            try:
                func = getattr(benchmarkfcns, fname)
                
                # Get function properties
                bounds = getattr(func, 'bounds', [[-100, 100]])
                dim = getattr(func, 'dim', 30)
                min_value = getattr(func, 'min_value', 0.0)
                
                # Determine if function is unimodal based on user specifications
                is_unimodal = fname in [
                    'ackleyn2', 'bohachevskyn1', 'booth', 'brent', 'brown', 'dropwave', 'exponential',
                    'powellsum', 'ridge', 'schaffern1', 'schaffern2', 'schaffern3', 'schaffern4',
                    'schwefel220', 'schwefel221', 'schwefel222', 'schwefel223', 'sphere', 'sumsquares',
                    'threehumpcamel', 'trid', 'xinsheyangn3'
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
        """Get function information by name
        
        Args:
            name (str): Name of the benchmark function
            
        Returns:
            dict: Dictionary containing function properties (func, bounds, dim, min_value, is_unimodal)
            
        Raises:
            ValueError: If the function name is not available
        """
        if name not in self.functions:
            raise ValueError(f"Function {name} not available")
        return self.functions[name]
    
    def evaluate(self, name, x):
        """Evaluate a benchmark function at given point(s)
        
        Args:
            name (str): Name of the benchmark function
            x (numpy.ndarray): Input point(s) for evaluation
            
        Returns:
            float: Function value at the given point(s)
        """
        func_info = self.get_function(name)
        
        # Handle functions that require 2D input (check dimension requirements)
        required_dim = self.get_required_dimension(name)
        if required_dim == 2:
            # These functions require 2D array (1 row, n columns) where n is the dimension
            if x.ndim == 1:
                x = x.reshape(1, -1)
        
        result = func_info['func'](x)
        # Ensure we return a scalar value by summing component-wise results
        if hasattr(result, '__len__') and len(result) > 1:
            return np.sum(result)
        # Ensure we return a scalar, not an array
        if hasattr(result, 'item'):
            return result.item()
        return result
    
    def get_bounds(self, name, dim=None):
        """Get bounds for a benchmark function
        
        Args:
            name (str): Name of the benchmark function
            dim (int, optional): Dimension for bounds. If None, uses function's default dimension
            
        Returns:
            numpy.ndarray: Array of bounds for each dimension
        """
        func_info = self.get_function(name)
        if dim is not None:
            # Return bounds for specified dimension
            return np.tile(func_info['bounds'][0], (dim, 1))
        return np.tile(func_info['bounds'][0], (func_info['dim'], 1))
    
    def get_function_dimension_requirements(self):
        """Get dimension requirements for functions that have restrictions
        
        Returns:
            dict: Dictionary mapping function names to their required dimensions
        """
        return {
            # 1D functions
            'forrester': 1,
            'gramacylee': 1,
            
            # 2D functions
            'ackleyn2': 2,
            'ackleyn3': 2,
            'adjiman': 2,
            'bartelsconn': 2,
            'beale': 2,
            'bird': 2,
            'bohachevskyn1': 2,
            'bohachevskyn2': 2,
            'booth': 2,
            'brent': 2,
            'bukinn6': 2,
            'carromtable': 2,
            'crossintray': 2,
            'deckersaarts': 2,
            'dropwave': 2,
            'easom': 2,
            'eggcrate': 2,
            'elattarvidyasagardutta': 2,
            'goldsteinprice': 2,
            'himmelblau': 2,
            'holdertable': 2,
            'keane': 2,
            'leon': 2,
            'levin13': 2,
            'matyas': 2,
            'mccormick': 2,
            'schaffern1': 2,
            'schaffern2': 2,
            'schaffern3': 2,
            'schaffern4': 2,
            'threehumpcamel': 2,
            
            # 3D functions
            'wolfe': 3,
            
            # Note: Functions not listed here are n-dimensional (can work with any dimension)
        }
    
    def get_required_dimension(self, name):
        """Get the required dimension for a function, or None if any dimension is supported
        
        Args:
            name (str): Name of the benchmark function
            
        Returns:
            int or None: Required dimension if function has restrictions, None if any dimension is supported
        """
        dimension_requirements = self.get_function_dimension_requirements()
        return dimension_requirements.get(name, None)
    
    def is_dimension_compatible(self, name, dim):
        """Check if a function supports the given dimension
        
        Args:
            name (str): Name of the benchmark function
            dim (int): Dimension to check compatibility for
            
        Returns:
            bool: True if function supports the given dimension, False otherwise
        """
        required_dim = self.get_required_dimension(name)
        if required_dim is None:
            return True  # Function supports any dimension
        return dim == required_dim
    
    def get_reasonable_bounds(self, name, dim=None):
        """Get more reasonable bounds for optimization algorithms
        
        Args:
            name (str): Name of the benchmark function
            dim (int, optional): Dimension for bounds. If None, uses function's default dimension
            
        Returns:
            numpy.ndarray: Array of reasonable bounds for optimization
        """
        # Define reasonable bounds for common functions
        reasonable_bounds = {
            'sphere': [-5, 5],
            'ackley': [-32, 32],
            'ackleyn2': [-32, 32],
            'ackleyn3': [-32, 32],
            'ackleyn4': [-32, 32],
            'rastrigin': [-5.12, 5.12],
            'rosenbrock': [-5, 10],
            'griewank': [-600, 600],
            'schwefel': [-500, 500],
            'schwefel220': [-100, 100],
            'schwefel221': [-100, 100],
            'schwefel222': [-100, 100],
            'schwefel223': [-100, 100],
            'sumsquares': [-10, 10],
            'zakharov': [-5, 10],
            'dixon_price': [-10, 10],
            'levin13': [-10, 10],
            'powellsum': [-1, 1],
            'quartic': [-1.28, 1.28],
            'salomon': [-100, 100],
            'schaffern1': [-100, 100],
            'schaffern2': [-100, 100],
            'schaffern3': [-100, 100],
            'schaffern4': [-100, 100],
            'bohachevskyn1': [-100, 100],
            'bohachevskyn2': [-100, 100],
            'dropwave': [-5.12, 5.12],
            'easom': [-100, 100],
            'matyas': [-10, 10],
            'mccormick': [-1.5, 4],
            'beale': [-4.5, 4.5],
            'goldsteinprice': [-2, 2],
            'booth': [-10, 10],
            'bukinn2': [-15, -5],
            'bukinn4': [-15, -5],
            'bukinn6': [-15, -5],
            'threehumpcamel': [-5, 5],
            'holdertable': [-10, 10],
            'himmelblau': [-5, 5],
            'leon': [-1.2, 1.2],
            'eggholder': [-512, 512],
            'crossintray': [-10, 10],
            'shubert': [-10, 10],
            'shubertn3': [-10, 10],
            'shubertn4': [-10, 10],
            'vincent': [0.25, 10],
            'wolfe': [0, 2],
            'xinsheyangn1': [-2*np.pi, 2*np.pi],
            'xinsheyangn2': [-2*np.pi, 2*np.pi],
            'xinsheyangn3': [-2*np.pi, 2*np.pi],
            'xinsheyangn4': [-2*np.pi, 2*np.pi],
        }
        
        if name in reasonable_bounds:
            bounds = reasonable_bounds[name]
            if dim is not None:
                return np.array([bounds for _ in range(dim)])
            else:
                func_info = self.get_function(name)
                return np.array([bounds for _ in range(func_info['dim'])])
        else:
            # Fall back to original bounds but scaled down
            original_bounds = self.get_bounds(name, dim)
            # Scale down by factor of 10
            scaled_bounds = original_bounds / 10
            return scaled_bounds
    
    def get_dimension(self, name):
        """Get the default dimension for a function
        
        Args:
            name (str): Name of the benchmark function
            
        Returns:
            int: Default dimension of the function
        """
        return self.get_function(name)['dim']
    
    def get_min_value(self, name):
        """Get the minimum value for a function
        
        Args:
            name (str): Name of the benchmark function
            
        Returns:
            float: Minimum value of the function
        """
        return self.get_function(name)['min_value']
    
    def is_unimodal(self, name):
        """Check if a function is unimodal
        
        Args:
            name (str): Name of the benchmark function
            
        Returns:
            bool: True if function is unimodal, False if multimodal
        """
        return self.get_function(name)['is_unimodal']
    
    def get_all_functions(self):
        """Get list of all available function names
        
        Returns:
            list: List of all available benchmark function names
        """
        return list(self.functions.keys())
    
    def get_unimodal_functions(self):
        """Get list of all unimodal function names
        
        Returns:
            list: List of unimodal benchmark function names
        """
        return [f for f in self.functions if self.functions[f]['is_unimodal']]
    
    def get_multimodal_functions(self):
        """Get list of all multimodal function names
        
        Returns:
            list: List of multimodal benchmark function names
        """
        return [f for f in self.functions if not self.functions[f]['is_unimodal']]
    
    def get_functions_by_dimension_support(self):
        """Get functions categorized by dimension support
        
        Returns:
            dict: Dictionary with 'any_dimension' and 'two_dimension_only' keys containing function lists
        """
        dimension_requirements = self.get_function_dimension_requirements()
        
        any_dimension = []
        two_dimension = []
        
        for func_name in self.functions:
            if func_name in dimension_requirements:
                two_dimension.append(func_name)
            else:
                any_dimension.append(func_name)
        
        return {
            'any_dimension': any_dimension,
            'two_dimension_only': two_dimension
        }
    
    def get_compatible_functions(self, dim):
        """Get all functions that support the given dimension
        
        Args:
            dim (int): Dimension to check compatibility for
            
        Returns:
            list: List of function names that support the given dimension
        """
        dimension_requirements = self.get_function_dimension_requirements()
        
        compatible = []
        for func_name in self.functions:
            required_dim = dimension_requirements.get(func_name, None)
            if required_dim is None or required_dim == dim:
                compatible.append(func_name)
        
        return compatible
    
    def get_function_info(self, name):
        """Get complete information about a function for table generation
        
        Args:
            name (str): Name of the benchmark function
            
        Returns:
            dict: Dictionary containing function information (name, range, dim, fmin, type)
        """
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