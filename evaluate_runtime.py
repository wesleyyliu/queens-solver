import time
import os
import matplotlib.pyplot as plt
import numpy as np
from pycosat_solver import PycosatSolver
from cpsat_solver import CpSatSolver
from backtrack_solver import BacktrackSolver
from generate_puzzle import convert_txt_line_to_board

def evaluate_solvers(solvers, puzzle_dir="puzzles", sizes=None):
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
            
        with open(puzzle_file, "r") as f:
            puzzle_lines = f.readlines()
        
        # Run solver on puzzle
        for i, line in enumerate(puzzle_lines):
            board = convert_txt_line_to_board(line)
            
            times = {}
            
            solutions = []

            # Compute solve time
            for solver_name, solver_class in solvers.items():
                solver = solver_class(board)
                start_time = time.time()
                solution = solver.solve()
                solve_time = time.time() - start_time
                results[solver_name][size].append(solve_time)
                times[solver_name] = solve_time
                solutions.append(solution)

            # Check all pairs of solutions and make sure they are the same
            for j, solution in enumerate(solutions):
                for k, other_solution in enumerate(solutions):
                    if solution != other_solution:
                        print("\n--- MISMATCH DETECTED ---")
                        print(f"Puzzle index: {i+1}")
                        print("Board:")
                        for row in board:
                            print(row)
                        print(f"Solution from {list(solvers.keys())[j]}:")
                        for row in solution or []:
                            print(row)
                        print(f"Solution from {list(solvers.keys())[k]}:")
                        for row in other_solution or []:
                            print(row)
                        print(f"Times: {times}")
                    assert solution == other_solution
            
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
        
        plt.errorbar(sizes, avg_times, yerr=std_times, fmt="o-", label=solver_name)
    
    plt.xlabel("Puzzle Size (n × n)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.title("Solver Performance vs Puzzle Size")
    plt.legend()
    plt.grid(True)
    
    # Use log scale if max time >= 10x min time
    all_times = [time for solver in solver_names for size in sizes for time in results[solver][size]]
    if max(all_times) / max(0.000001, min(all_times)) > 10:
        plt.yscale("log")
        plt.title("Solver Performance vs Puzzle Size (Log Scale)")
    
    plt.tight_layout()
    # Output directory
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/execution_time_vs_size.png")
    plt.close()


def plot_execution_time_difference(results, solver1, solver2):
    """
    Plot runtime difference between two solvers
    
    Args:
        results: Dictionary with results from evaluate_solvers
        solver1: First solver name
        solver2: Second solver name
    """
    sizes = results["sizes"]
    
    if solver1 not in results or solver2 not in results:
        print(f"Error: One solver missing from results")
        return
    
    plt.figure(figsize=(10, 6))
    
    differences = []
    std_diffs = []
    
    for size in sizes:
        diffs = [results[solver2][size][i] - results[solver1][size][i] 
                for i in range(len(results[solver1][size]))]
        
        differences.append(np.mean(diffs))
        std_diffs.append(np.std(diffs))
    
    # Create bar chart
    plt.bar(sizes, differences, yerr=std_diffs, alpha=0.7)
    
    # Baseline 0 horizontal line
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    
    plt.xlabel("Puzzle Size (n × n)")
    plt.ylabel(f"Time Difference: {solver2} - {solver1} (seconds)")
    plt.title(f"Runtime Difference: {solver2} vs {solver1}")
    plt.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    # Output directory
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/solver_difference_{solver1}_vs_{solver2}.png")
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    solvers = {
        "PycosatSolver": PycosatSolver,
        "PycosatSolver (with heuristics)": lambda board: PycosatSolver(board, use_heuristic=True),
        "CpSatSolver": CpSatSolver,
        "CpSatSolver (with heuristics)": lambda board: CpSatSolver(board, use_heuristic=True)
    }
    
    results = evaluate_solvers(solvers, sizes=list(range(5,16)))
    
    # Create visualizations
    plot_execution_time_vs_size(results)
    solver_names = list(solvers.keys())
    for i, solver1 in enumerate(solver_names):
        for solver2 in solver_names[i+1:]:
            plot_execution_time_difference(results, solver1, solver2)
    print("Evaluation done.")