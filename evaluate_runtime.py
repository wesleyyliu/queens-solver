import time
import os
import matplotlib.pyplot as plt
import numpy as np
from pycosat_solver import PycosatSolver
from backtrack_solver import BacktrackSolver
from generate_puzzle import convert_txt_line_to_board

def evaluate_solvers(solvers, puzzle_dir="puzzles", sizes=None, num_samples=50):
    """
    Benchmark solvers on puzzles of different sizes.
    
    Args:
        solvers: Dictionary of solvers to benchmark {"solver_name": solver_class}
        puzzle_dir: Directory containing puzzle files
        sizes: List of puzzle sizes to benchmark
        num_samples: Number of puzzles to sample for each size
    
    Returns:
        Dictionary with benchmark results
    """
    if sizes is None:
        # Get all sizes from puzzle files
        sizes = []
        for filename in os.listdir(puzzle_dir):
            if filename.startswith("size_") and filename.endswith(".txt"):
                size = int(filename.split("_")[1].split(".")[0])
                sizes.append(size)
        sizes.sort()
    
    results = {
        "sizes": sizes,
    }
    
    # Initialize results for each solver
    for solver_name in solvers:
        results[solver_name] = {size: [] for size in sizes}
    
    for size in sizes:
        print(f"Evaluating runtime of puzzles of size {size}x{size}...")
        
        # Load puzzles
        puzzle_file = os.path.join(puzzle_dir, f"size_{size}.txt")
        if not os.path.exists(puzzle_file):
            print(f"No puzzles found for size {size}. Skipping.")
            continue
            
        with open(puzzle_file, 'r') as f:
            puzzle_lines = f.readlines()
        
        # Sample puzzles randomly if we exceed the number of samples we want to look at
        if len(puzzle_lines) > num_samples:
            puzzle_lines = np.random.choice(puzzle_lines, num_samples, replace=False)
        
        # Run solver on puzzle
        for i, line in enumerate(puzzle_lines):
            board = convert_txt_line_to_board(line)
            
            times = {}
            
            # Compute solve time
            for solver_name, solver_class in solvers.items():
                solver = solver_class(board)
                start_time = time.time()
                solver.solve()
                solve_time = time.time() - start_time
                results[solver_name][size].append(solve_time)
                times[solver_name] = solve_time
            
            # Get times for each solver and print it out for each puzzle
            time_strings = [f"{name}: {time:.4f}s" for name, time in times.items()]
            print(f"  Puzzle {i+1}/{len(puzzle_lines)}: {', '.join(time_strings)}")
    
    return results

def plot_execution_time_vs_size(results):
    """Plot average execution time vs puzzle size for all solvers"""
    sizes = results["sizes"]
    solver_names = [key for key in results.keys() if key != "sizes"]
    
    plt.figure(figsize=(10, 6))
    
    for solver_name in solver_names:
        # Calculate avg and stdev of times
        avg_times = [np.mean(results[solver_name][size]) for size in sizes]
        std_times = [np.std(results[solver_name][size]) for size in sizes]
        
        plt.errorbar(sizes, avg_times, yerr=std_times, fmt='o-', label=solver_name)
    
    plt.xlabel('Puzzle Size (n × n)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Solver Performance vs Puzzle Size')
    plt.legend()
    plt.grid(True)
    
    # Use log scale if max time >= 10x min time
    all_times = [time for solver in solver_names for size in sizes for time in results[solver][size]]
    if max(all_times) / max(0.000001, min(all_times)) > 10:
        plt.yscale('log')
        plt.title('Solver Performance vs Puzzle Size (Log Scale)')
    
    plt.tight_layout()
    plt.savefig('execution_time_vs_size.png')
    plt.close()

if __name__ == "__main__":
    solvers = {
        "PycosatSolver": PycosatSolver,
        "BacktrackSolver": BacktrackSolver,
    }
    
    results = evaluate_solvers(solvers, sizes=list(range(5,8)), num_samples=500)
    
    # Create visualizations
    plot_execution_time_vs_size(results)
    
    print("Evaluation done.")