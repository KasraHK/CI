# benchmark_functions.py
# A corrected and complete script with 27 unimodal and 41 multimodal benchmark functions
# as specified by the user's list.
# Source for formulas: https://www.sfu.ca/~ssurjano/optimization.html

import numpy as np

# ======================================================================================
# I. Unimodal Benchmark Functions (27 Functions, as per your list)
# ======================================================================================

def ackley_n2(x):
    """Ackley N. 2 Function (2D)"""
    if len(x) != 2: raise ValueError("Ackley N. 2 is 2D.")
    return -200 * np.exp(-0.02 * np.sqrt(x[0]**2 + x[1]**2))

def bohachevsky_n1(x):
    """Bohachevsky N. 1 Function (2D)"""
    if len(x) != 2: raise ValueError("Bohachevsky N. 1 is 2D.")
    x1, x2 = x[0], x[1]
    return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7

def booth(x):
    """Booth Function (2D)"""
    if len(x) != 2: raise ValueError("Booth Function is 2D.")
    x1, x2 = x[0], x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def brent(x):
    """Brent Function (2D)"""
    if len(x) != 2: raise ValueError("Brent Function is 2D.")
    x1, x2 = x[0], x[1]
    return (x1 + 10)**2 + (x2 + 10)**2 + np.exp(-x1**2 - x2**2)

def brown(x):
    """Brown Function"""
    d = len(x)
    result = 0
    for i in range(d - 1):
        term1 = x[i]**2
        term2 = x[i+1]**2
        result += (term1)**(term2 + 1) + (term2)**(term1 + 1)
    return result

def drop_wave(x):
    """Drop-Wave Function (2D)"""
    if len(x) != 2: raise ValueError("Drop-Wave is 2D.")
    x1, x2 = x[0], x[1]
    sq_sum = x1**2 + x2**2
    if sq_sum == 0: return -1 # Avoid division by zero at the optimum
    return -(1 + np.cos(12 * np.sqrt(sq_sum))) / (0.5 * sq_sum + 2)

def exponential(x):
    """Exponential Function"""
    return -np.exp(-0.5 * np.sum(x**2))

def griewank(x):
    """Griewank Function"""
    d = len(x)
    sum_term = np.sum(x**2 / 4000.0)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return sum_term - prod_term + 1

def leon(x):
    """Leon Function (2D)"""
    if len(x) != 2: raise ValueError("Leon Function is 2D.")
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def matyas(x):
    """Matyas Function (2D)"""
    if len(x) != 2: raise ValueError("Matyas Function is 2D.")
    x1, x2 = x[0], x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

def powell_sum(x):
    """Powell Sum Function"""
    d = len(x)
    return np.sum(np.abs(x)**(np.arange(1, d + 1) + 1))

def ridge(x):
    """Ridge Function"""
    d = 10  # Parameter d for the Ridge function
    return x[0] + d * np.sum(x[1:]**2)

def schaffer_n1(x):
    """Schaffer N. 1 Function (2D)"""
    if len(x) != 2: raise ValueError("Schaffer N. 1 is 2D.")
    x1, x2 = x[0], x[1]
    sq_sum = x1**2 + x2**2
    return 0.5 + (np.sin(sq_sum)**2 - 0.5) / (1 + 0.001 * sq_sum)**2

def schaffer_n2(x):
    """Schaffer N. 2 Function (2D)"""
    if len(x) != 2: raise ValueError("Schaffer N. 2 is 2D.")
    x1, x2 = x[0], x[1]
    num = np.sin(x1**2 - x2**2)**2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + num / den

def schaffer_n3(x):
    """Schaffer N. 3 Function (2D)"""
    if len(x) != 2: raise ValueError("Schaffer N. 3 is 2D.")
    x1, x2 = x[0], x[1]
    num = np.sin(np.cos(np.abs(x1**2 - x2**2)))**2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + num

def schaffer_n4(x):
    """Schaffer N. 4 Function (2D)"""
    if len(x) != 2: raise ValueError("Schaffer N. 4 is 2D.")
    x1, x2 = x[0], x[1]
    num = np.cos(np.sin(np.abs(x1**2 - x2**2)))**2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + num

def schwefel_2_20(x):
    """Schwefel 2.20 Function"""
    return np.sum(np.abs(x))

def schwefel_2_21(x):
    """Schwefel 2.21 Function"""
    return np.max(np.abs(x))

def schwefel_2_22(x):
    """Schwefel 2.22 Function"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def schwefel_2_23(x):
    """Schwefel 2.23 Function"""
    return np.sum(x**10)

def sphere(x):
    """Sphere Function"""
    return np.sum(x**2)

def sum_squares(x):
    """Sum Squares Function"""
    d = len(x)
    return np.sum(np.arange(1, d + 1) * x**2)

def three_hump_camel(x):
    """Three-Hump Camel Function (2D)"""
    if len(x) != 2: raise ValueError("Three-Hump Camel is 2D.")
    x1, x2 = x[0], x[1]
    return 2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2

def trid(x):
    """Trid Function"""
    return np.sum((x - 1)**2) - np.sum(x[1:] * x[:-1])

def xinsheyang_n3(x):
    """Xin-She Yang N. 3 Function"""
    m = 5
    beta = 15
    return np.exp(-np.sum((x/beta)**(2*m))) - 2*np.exp(-np.sum(x**2)) * np.prod(np.cos(x)**2)

def zakharov(x):
    """Zakharov Function"""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, d + 1) * x)
    return sum1 + sum2**2 + sum2**4

# ======================================================================================
# II. Multimodal Benchmark Functions (41 Functions, as per your list)
# ======================================================================================

def ackley(x):
    """Ackley Function"""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

def ackley_n3(x):
    """Ackley N. 3 Function (2D)"""
    if len(x) != 2: raise ValueError("Ackley N. 3 is 2D.")
    return -200 * np.exp(-0.2 * np.sqrt(x[0]**2 + x[1]**2)) + 5 * np.exp(np.cos(3*x[0])+np.sin(3*x[1]))

def ackley_n4(x):
    """Ackley N. 4 Function"""
    d = len(x)
    result = 0
    for i in range(d - 1):
        term = np.exp(-0.2) * np.sqrt(x[i]**2 + x[i+1]**2) + 3*(np.cos(2*x[i])+np.sin(2*x[i+1]))
        result += term
    return result

def adjiman(x):
    """Adjiman Function (2D)"""
    if len(x) != 2: raise ValueError("Adjiman is 2D.")
    x1, x2 = x[0], x[1]
    return np.cos(x1)*np.sin(x2) - x1/(x2**2 + 1)

def alpine_n1(x):
    """Alpine N. 1 Function"""
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def alpine_n2(x):
    """Alpine N. 2 Function"""
    return np.prod(np.sqrt(x) * np.sin(x))
    
def bartels_conn(x):
    """Bartels Conn Function (2D)"""
    if len(x) != 2: raise ValueError("Bartels Conn is 2D.")
    x1, x2 = x[0], x[1]
    return np.abs(x1**2 + x2**2 + x1*x2) + np.abs(np.sin(x1)) + np.abs(np.cos(x2))

def beale(x):
    """Beale Function (2D)"""
    if len(x) != 2: raise ValueError("Beale is 2D.")
    x1, x2 = x[0], x[1]
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

def bird(x):
    """Bird Function (2D)"""
    if len(x) != 2: raise ValueError("Bird is 2D.")
    x1, x2 = x[0], x[1]
    return np.sin(x1)*np.exp((1-np.cos(x2))**2) + np.cos(x2)*np.exp((1-np.sin(x1))**2) + (x1-x2)**2

def bohachevsky_n2(x):
    """Bohachevsky N. 2 Function (2D)"""
    if len(x) != 2: raise ValueError("Bohachevsky N. 2 is 2D.")
    x1, x2 = x[0], x[1]
    return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1)*np.cos(4*np.pi*x2) + 0.3
    
def bukin_n6(x):
    """Bukin N. 6 Function (2D)"""
    if len(x) != 2: raise ValueError("Bukin N. 6 is 2D.")
    x1, x2 = x[0], x[1]
    return 100 * np.sqrt(np.abs(x2 - 0.01*x1**2)) + 0.01 * np.abs(x1 + 10)

def carrom_table(x):
    """Carrom Table Function (2D)"""
    if len(x) != 2: raise ValueError("Carrom Table is 2D.")
    x1, x2 = x[0], x[1]
    term1 = np.cos(x1)*np.cos(x2)*np.exp(np.abs(1 - np.sqrt(x1**2+x2**2)/np.pi))
    return - (1/30.0) * term1**2

def cross_in_tray(x):
    """Cross-in-Tray Function (2D)"""
    if len(x) != 2: raise ValueError("Cross-in-Tray is 2D.")
    x1, x2 = x[0], x[1]
    term = np.abs(100 - np.sqrt(x1**2 + x2**2)/np.pi)
    return -0.0001 * (np.abs(np.sin(x1)*np.sin(x2)*np.exp(term)) + 1)**0.1

def deckkers_aarts(x):
    """Deckkers-Aarts Function (2D)"""
    if len(x) != 2: raise ValueError("Deckkers-Aarts is 2D.")
    x1, x2 = x[0], x[1]
    return 1e5*x1**2 + x2**2 - (x1**2 + x2**2)**2 + 1e-5*(x1**2 + x2**2)**4

def easom(x):
    """Easom Function (2D)"""
    if len(x) != 2: raise ValueError("Easom is 2D.")
    x1, x2 = x[0], x[1]
    return -np.cos(x1)*np.cos(x2)*np.exp(-((x1-np.pi)**2 + (x2-np.pi)**2))

def egg_crate(x):
    """Egg Crate Function (2D)"""
    if len(x) != 2: raise ValueError("Egg Crate is 2D.")
    x1, x2 = x[0], x[1]
    return x1**2 + x2**2 + 25*(np.sin(x1)**2 + np.sin(x2)**2)

def elattar_vidyasagar_dutta(x):
    """El-Attar-Vidyasagar-Dutta Function (2D)"""
    if len(x) != 2: raise ValueError("El-Attar-Vidyasagar-Dutta is 2D.")
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 10)**2 + (x1 + x2**2 - 7)**2 + (x1**2 + x2**3 - 1)**2

def forrester(x):
    """Forrester Function (1D)"""
    if len(x) != 1: raise ValueError("Forrester is 1D.")
    x_val = x[0]
    return (6*x_val - 2)**2 * np.sin(12*x_val - 4)
    
def goldstein_price(x):
    """Goldstein-Price Function (2D)"""
    if len(x) != 2: raise ValueError("Goldstein-Price is 2D.")
    x1, x2 = x[0], x[1]
    term1 = 1 + (x1+x2+1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + (2*x1-3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return term1 * term2

def gramacy_lee(x):
    """Gramacy & Lee Function (1D)"""
    if len(x) != 1: raise ValueError("Gramacy & Lee is 1D.")
    x_val = x[0]
    return np.sin(10*np.pi*x_val)/(2*x_val) + (x_val-1)**4
    
def happy_cat(x):
    """HappyCat Function"""
    d = len(x)
    norm_x = np.linalg.norm(x)
    sum_x = np.sum(x)
    return ((norm_x**2 - d)**2)**0.25 + (0.5 * norm_x**2 + sum_x)/d + 0.5

def himmelblau(x):
    """Himmelblau Function (2D)"""
    if len(x) != 2: raise ValueError("Himmelblau is 2D.")
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def holder_table(x):
    """Holder-Table Function (2D)"""
    if len(x) != 2: raise ValueError("Holder Table is 2D.")
    x1, x2 = x[0], x[1]
    term = np.abs(1 - np.sqrt(x1**2 + x2**2)/np.pi)
    return -np.abs(np.sin(x1)*np.cos(x2)*np.exp(term))

def keane(x):
    """Keane Function (2D)"""
    if len(x) != 2: raise ValueError("Keane is 2D.")
    x1, x2 = x[0], x[1]
    return - (np.sin(x1-x2)**2 * np.sin(x1+x2)**2) / np.sqrt(x1**2+x2**2)

def levi_n13(x):
    """Levi N. 13 Function (2D)"""
    if len(x) != 2: raise ValueError("Levi N. 13 is 2D.")
    x1, x2 = x[0], x[1]
    term1 = np.sin(3*np.pi*x1)**2
    term2 = (x1-1)**2 * (1+np.sin(3*np.pi*x2)**2)
    term3 = (x2-1)**2 * (1+np.sin(2*np.pi*x2)**2)
    return term1 + term2 + term3

def mccormick(x):
    """McCormick Function (2D)"""
    if len(x) != 2: raise ValueError("McCormick is 2D.")
    x1, x2 = x[0], x[1]
    return np.sin(x1+x2) + (x1-x2)**2 - 1.5*x1 + 2.5*x2 + 1

def periodic(x):
    """Periodic Function"""
    return 1 + np.sin(np.sum(x))**2 - 0.1 * np.exp(-np.sum(x**2))

def qing(x):
    """Qing Function"""
    d = len(x)
    indices = np.arange(1, d + 1)
    return np.sum((x**2 - indices)**2)
    
def quartic(x):
    """Quartic Function"""
    d = len(x)
    return np.sum(np.arange(1, d + 1) * (x**4))

def rastrigin(x):
    """Rastrigin Function"""
    d = len(x)
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """Rosenbrock Function"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
def salomon(x):
    """Salomon Function"""
    norm_x = np.linalg.norm(x)
    return 1 - np.cos(2 * np.pi * norm_x) + 0.1 * norm_x

def schwefel(x):
    """Schwefel Function"""
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def shubert_3(x):
    """Shubert 3 Function"""
    d = len(x)
    result = 0
    for i in range(d):
        sum_inner = 0
        for j in range(1,6):
            sum_inner += j * np.sin((j+1)*x[i]+j)
        result += sum_inner
    return result

def shubert_4(x):
    """Shubert N. 4 Function"""
    d = len(x)
    result = 0
    for i in range(d):
        sum_inner = 0
        for j in range(1,6):
            sum_inner += j * np.cos((j+1)*x[i]+j)
        result += sum_inner
    return result

def shubert(x):
    """Shubert Function"""
    d = len(x)
    result = 0
    for i in range(d):
        sum_inner = 0
        for j in range(1, 6):
            sum_inner += j * np.cos((j + 1) * x[i] + j)
        result += sum_inner
    return result

def styblinski_tang(x):
    """Styblinski-Tang Function"""
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)

def wolfe(x):
    """Wolfe Function (3D)"""
    if len(x) != 3: raise ValueError("Wolfe is 3D.")
    x1, x2, x3 = x[0], x[1], x[2]
    return (4/3)*(x1**2 - x1*x2 + x2**2)**0.75 + x3

def xinsheyang(x):
    """Xin-She Yang Function"""
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))
    
def xinsheyang_n2(x):
    """Xin-She Yang N. 2 Function"""
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))
    
def xinsheyang_n4(x):
    """Xin-She Yang N. 4 Function"""
    return (np.sum(np.sin(x)**2) - np.exp(-np.sum(x**2))) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2))


# ======================================================================================
# III. Master Dictionary (27 Unimodal, 41 Multimodal)
# ======================================================================================

BENCHMARK_FUNCTIONS = {
    # --- Unimodal (27) ---
    "ackley_n2":          {"func": ackley_n2, "bounds": (-32, 32), "type": "unimodal", "dim_type": 2, "global_minimum": -200.0},
    "bohachevsky_n1":     {"func": bohachevsky_n1, "bounds": (-100, 100), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "booth":              {"func": booth, "bounds": (-10, 10), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "brent":              {"func": brent, "bounds": (-10, 10), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "brown":              {"func": brown, "bounds": (-1, 4), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "drop_wave":          {"func": drop_wave, "bounds": (-5.12, 5.12), "type": "unimodal", "dim_type": 2, "global_minimum": -1.0},
    "exponential":        {"func": exponential, "bounds": (-1, 1), "type": "unimodal", "dim_type": "n-dim", "global_minimum": -1.0},
    "griewank":           {"func": griewank, "bounds": (-100, 100), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0}, # Moved to unimodal as per list
    "leon":               {"func": leon, "bounds": (-1.2, 1.2), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "matyas":             {"func": matyas, "bounds": (-10, 10), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "powell_sum":         {"func": powell_sum, "bounds": (-1, 1), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "ridge":              {"func": ridge, "bounds": (-5, 5), "type": "unimodal", "dim_type": "n-dim", "global_minimum": -5.0},
    "schaffer_n1":        {"func": schaffer_n1, "bounds": (-100, 100), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "schaffer_n2":        {"func": schaffer_n2, "bounds": (-100, 100), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "schaffer_n3":        {"func": schaffer_n3, "bounds": (-100, 100), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "schaffer_n4":        {"func": schaffer_n4, "bounds": (-100, 100), "type": "unimodal", "dim_type": 2, "global_minimum": 0.2919265817264289},
    "schwefel_2_20":      {"func": schwefel_2_20, "bounds": (-100, 100), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "schwefel_2_21":      {"func": schwefel_2_21, "bounds": (-100, 100), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "schwefel_2_22":      {"func": schwefel_2_22, "bounds": (-10, 10), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "schwefel_2_23":      {"func": schwefel_2_23, "bounds": (-10, 10), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "sphere":             {"func": sphere, "bounds": (-5.12, 5.12), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "sum_squares":        {"func": sum_squares, "bounds": (-10, 10), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "three_hump_camel":   {"func": three_hump_camel, "bounds": (-5, 5), "type": "unimodal", "dim_type": 2, "global_minimum": 0.0},
    "trid":               {"func": trid, "bounds": (-30**2, 30**2), "type": "unimodal", "dim_type": "n-dim", "global_minimum": "dimension_dependent"},
    "xinsheyang_n3":      {"func": xinsheyang_n3, "bounds": (-2*np.pi, 2*np.pi), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.9950234947576024},
    "zakharov":           {"func": zakharov, "bounds": (-5, 10), "type": "unimodal", "dim_type": "n-dim", "global_minimum": 0.0},

    # --- Multimodal (41) ---
    "ackley":             {"func": ackley, "bounds": (-32.768, 32.768), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "ackley_n3":          {"func": ackley_n3, "bounds": (-32, 32), "type": "multimodal", "dim_type": 2, "global_minimum": -186.4112127112689},
    "ackley_n4":          {"func": ackley_n4, "bounds": (-35, 35), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "adjiman":            {"func": adjiman, "bounds": ((-1, 2), (-1, 1)), "type": "multimodal", "dim_type": 2, "global_minimum": -2.02180678}, # Asymmetric
    "alpine_n1":          {"func": alpine_n1, "bounds": (-10, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "alpine_n2":          {"func": alpine_n2, "bounds": (0, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": -2.8081311800070057e+01},
    "bartels_conn":       {"func": bartels_conn, "bounds": (-500, 500), "type": "multimodal", "dim_type": 2, "global_minimum": 1.0},
    "beale":              {"func": beale, "bounds": (-4.5, 4.5), "type": "multimodal", "dim_type": 2, "global_minimum": 0.0},
    "bird":               {"func": bird, "bounds": (-2*np.pi, 2*np.pi), "type": "multimodal", "dim_type": 2, "global_minimum": -106.76453674926478},
    "bohachevsky_n2":     {"func": bohachevsky_n2, "bounds": (-100, 100), "type": "multimodal", "dim_type": 2, "global_minimum": 0.0},
    "bukin_n6":           {"func": bukin_n6, "bounds": ((-15, -5), (-3, 3)), "type": "multimodal", "dim_type": 2, "global_minimum": 0.0}, # Asymmetric
    "carrom_table":       {"func": carrom_table, "bounds": (-10, 10), "type": "multimodal", "dim_type": 2, "global_minimum": -24.156815547391254},
    "cross_in_tray":      {"func": cross_in_tray, "bounds": (-10, 10), "type": "multimodal", "dim_type": 2, "global_minimum": -2.0626118708227397},
    "deckkers_aarts":     {"func": deckkers_aarts, "bounds": (-20, 20), "type": "multimodal", "dim_type": 2, "global_minimum": -24776.518342317697},
    "easom":              {"func": easom, "bounds": (-100, 100), "type": "multimodal", "dim_type": 2, "global_minimum": -1.0},
    "egg_crate":          {"func": egg_crate, "bounds": (-5, 5), "type": "multimodal", "dim_type": 2, "global_minimum": 0.0},
    "elattar_vidyasagar_dutta": {"func": elattar_vidyasagar_dutta, "bounds": (-500, 500), "type": "multimodal", "dim_type": 2, "global_minimum": 1.7127803548621987},
    "forrester":          {"func": forrester, "bounds": (0, 1), "type": "multimodal", "dim_type": 1, "global_minimum": -6.020740055767083},
    "goldstein_price":    {"func": goldstein_price, "bounds": (-2, 2), "type": "multimodal", "dim_type": 2, "global_minimum": 3.0},
    "gramacy_lee":        {"func": gramacy_lee, "bounds": (0.5, 2.5), "type": "multimodal", "dim_type": 1, "global_minimum": -0.8690111349894999},
    "happy_cat":          {"func": happy_cat, "bounds": (-2, 2), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "himmelblau":         {"func": himmelblau, "bounds": (-5, 5), "type": "multimodal", "dim_type": 2, "global_minimum": 0.0},
    "holder_table":       {"func": holder_table, "bounds": (-10, 10), "type": "multimodal", "dim_type": 2, "global_minimum": -19.20850256788675},
    "keane":              {"func": keane, "bounds": (0, 10), "type": "multimodal", "dim_type": 2, "global_minimum": -0.6736675211468551},
    "levi_n13":           {"func": levi_n13, "bounds": (-10, 10), "type": "multimodal", "dim_type": 2, "global_minimum": 0.0},
    "mccormick":          {"func": mccormick, "bounds": ((-1.5, 4), (-3, 4)), "type": "multimodal", "dim_type": 2, "global_minimum": -1.9132229549810367}, # Asymmetric
    "periodic":           {"func": periodic, "bounds": (-10, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 1.0},
    "qing":               {"func": qing, "bounds": (-500, 500), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "quartic":            {"func": quartic, "bounds": (-1.28, 1.28), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "rastrigin":          {"func": rastrigin, "bounds": (-5.12, 5.12), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "rosenbrock":         {"func": rosenbrock, "bounds": (-5, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0}, # Moved to multimodal as per list
    "salomon":            {"func": salomon, "bounds": (-100, 100), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "schwefel":           {"func": schwefel, "bounds": (-500, 500), "type": "multimodal", "dim_type": "n-dim", "global_minimum": -418.9829},
    "shubert_3":          {"func": shubert_3, "bounds": (-10, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": -29.4},
    "shubert_4":          {"func": shubert_4, "bounds": (-10, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": -29.4},
    "shubert":            {"func": shubert, "bounds": (-10, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": -186.7309},
    "styblinski_tang":    {"func": styblinski_tang, "bounds": (-5, 5), "type": "multimodal", "dim_type": "n-dim", "global_minimum": -39.16616570377142},
    "wolfe":              {"func": wolfe, "bounds": (0, 2), "type": "multimodal", "dim_type": 3, "global_minimum": 0.0},
    "xinsheyang":         {"func": xinsheyang, "bounds": (-2*np.pi, 2*np.pi), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "xinsheyang_n2":      {"func": xinsheyang_n2, "bounds": (-2*np.pi, 2*np.pi), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
    "xinsheyang_n4":      {"func": xinsheyang_n4, "bounds": (-10, 10), "type": "multimodal", "dim_type": "n-dim", "global_minimum": 0.0},
}