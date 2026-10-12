import argparse
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from solvers.pycosat_solver import PycosatSolver
from solvers.cpsat_solver import CpSatSolver
from solvers.backtrack_solver import BacktrackSolver, RowBacktrackSolver
from generators.generate_puzzle import convert_txt_line_to_board
from solvers.base import is_valid_solution as verify_solution

ALL_SOLVERS = {
    "PycosatSolver": PycosatSolver,
    "PycosatSolver (with heuristics)": lambda board: PycosatSolver(board, use_heuristic=True),
    "CpSatSolver": CpSatSolver,
    "CpSatSolver (with heuristics)": lambda board: CpSatSolver(board, use_heuristic=True),
    "BacktrackSolver": BacktrackSolver,
    "RowBacktrackSolver": RowBacktrackSolver,
}

def time_solver(solver_class, board, repeats):
    """
    Time a solver on a board, returning the fastest of `repeats` runs and the solution.
    The minimum is the least noisy estimate, since noise (GC, OS scheduling) only ever adds time.
    """
    best_time = float("inf")
    solution = None
    for _ in range(repeats):
        solver = solver_class(board)
        start_time = time.perf_counter()
        solution = solver.solve()
        best_time = min(best_time, time.perf_counter() - start_time)
    return best_time, solution

def evaluate_solvers(solvers, puzzle_dir="puzzles", sizes=None, limit=None, repeats=1, max_sizes=None):
    """
    Benchmark solvers on puzzles of different sizes.

    Args:
        solvers: Dictionary of solvers to benchmark {"solver_name": solver_class}
        puzzle_dir: Directory containing puzzle files
        sizes: List of puzzle sizes to benchmark
        limit: Maximum number of puzzles to run for each size (None for all)
        repeats: Number of times to run each solver on each puzzle (fastest run is kept)
        max_sizes: Optional {"solver_name": max_size} to skip slow solvers on large puzzles

    Returns:
        Dictionary with benchmark results
    """
    max_sizes = max_sizes or {}

    if sizes is None:
        # Get all sizes from puzzle files
        sizes = []
        for filename in os.listdir(puzzle_dir):
            if filename.startswith("size_") and filename.endswith(".txt"):
                size = int(filename.split("_")[1].split(".")[0])
                sizes.append(size)
        sizes.sort()

    # Only keep sizes that have puzzle files so plots don't contain empty entries
    available_sizes = []
    for size in sizes:
        if os.path.exists(os.path.join(puzzle_dir, f"size_{size}.txt")):
            available_sizes.append(size)
        else:
            print(f"No puzzles found for size {size}. Skipping.")
    sizes = available_sizes

    results = {
        "sizes": sizes,
    }

    # Initialize results for each solver
    for solver_name in solvers:
        results[solver_name] = {size: [] for size in sizes}

    # Warm up each solver once so one-time costs (imports, native library loading) aren't timed
    if sizes:
        with open(os.path.join(puzzle_dir, f"size_{sizes[0]}.txt"), "r") as f:
            warmup_board = convert_txt_line_to_board(f.readline())
        for solver_class in solvers.values():
            solver_class(warmup_board).solve()

    for size in sizes:
        print(f"Evaluating runtime of puzzles of size {size}x{size}...")

        # Load puzzles
        puzzle_file = os.path.join(puzzle_dir, f"size_{size}.txt")
        with open(puzzle_file, "r") as f:
            puzzle_lines = [line for line in f.readlines() if line.strip()]
        if limit is not None:
            puzzle_lines = puzzle_lines[:limit]

        # Run solver on puzzle
        for i, line in enumerate(puzzle_lines):
            board = convert_txt_line_to_board(line)

            times = {}
            solutions = {}

            # Compute solve time, skipping solvers that are too slow for this size
            for solver_name, solver_class in solvers.items():
                if size > max_sizes.get(solver_name, size):
                    continue
                solve_time, solution = time_solver(solver_class, board, repeats)
                results[solver_name][size].append(solve_time)
                times[solver_name] = solve_time
                solutions[solver_name] = solution

            # Check every solution against the rules, then check all solvers agree (puzzles have a unique solution)
            reference_name = next(iter(solutions))
            for solver_name, solution in solutions.items():
                if not verify_solution(board, solution) or solution != solutions[reference_name]:
                    print("\n--- INVALID SOLUTION OR MISMATCH DETECTED ---")
                    print(f"Puzzle index: {i+1}")
                    print("Board:")
                    for row in board:
                        print(row)
                    for name, other_solution in solutions.items():
                        print(f"Solution from {name} (valid: {verify_solution(board, other_solution)}):")
                        for row in other_solution or []:
                            print(row)
                    print(f"Times: {times}")
                    raise AssertionError(f"{solver_name} produced an invalid or mismatched solution")

            # Get times for each solver and print it out for each puzzle
            time_strings = [f"{name}: {time * 1000:.3f}ms" for name, time in times.items()]
            print(f"  Puzzle {i+1}/{len(puzzle_lines)}: {', '.join(time_strings)}")

    return results

def print_summary(results):
    """Print median and interquartile range of solve times for each solver and size"""
    solver_names = [key for key in results.keys() if key != "sizes"]
    print("\nSummary (median [25th-75th percentile] in ms):")
    for solver_name in solver_names:
        print(f"  {solver_name}")
        for size in sizes_with_data(results, solver_name):
            times_ms = np.array(results[solver_name][size]) * 1000
            p25, median, p75 = np.percentile(times_ms, [25, 50, 75])
            print(f"    {size:>2}x{size:<2}: {median:8.3f} [{p25:.3f}-{p75:.3f}]  (n={len(times_ms)})")

def plot_execution_time_vs_size(results, log_scale=False):
    """
    Plot median execution time vs puzzle size for all solvers, with interquartile range error bars.
    Median/IQR is used rather than mean/stdev because solve times are right-skewed, and stdev bars
    can extend below zero (which breaks the log scale plot).
    """
    sizes = results["sizes"]
    solver_names = [key for key in results.keys() if key != "sizes"]

    plt.figure(figsize=(10, 6))

    for solver_name in solver_names:
        solver_sizes = sizes_with_data(results, solver_name)
        if not solver_sizes:
            continue
        p25, median, p75 = np.percentile([results[solver_name][size] for size in solver_sizes], [25, 50, 75], axis=1)
        yerr = [median - p25, p75 - median]
        plt.errorbar(solver_sizes, median, yerr=yerr, fmt="o-", capsize=3, label=solver_name)

    plt.xlabel("Puzzle Size (n × n)")
    plt.ylabel("Median Execution Time (seconds)")
    plt.title("Solver Performance vs Puzzle Size")
    plt.legend()
    plt.grid(True)

    # Use log scale if max time >= 10x min time
    all_times = [time for solver in solver_names for size in sizes for time in results[solver][size]]
    if log_scale and max(all_times) / max(0.000001, min(all_times)) > 10:
        plt.yscale("log")
        plt.title("Solver Performance vs Puzzle Size (Log Scale)")

    plt.tight_layout()
    # Output directory
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/execution_time_vs_size.png" if not log_scale else "figs/execution_time_vs_size_log.png")
    plt.close()


def plot_execution_time_percentage_difference(results, solver1, solver2):
    """
    Plot runtime percentage difference between two solvers.

    Uses the geometric mean of per-puzzle time ratios (solver2 / solver1), which is the correct average
    for ratios: the arithmetic mean of percentage differences is biased upwards, since a 2x slowdown
    counts as +100% but a 2x speedup only counts as -50%. Error bars show the interquartile range.

    Args:
        results: Dictionary with results from evaluate_solvers
        solver1: First solver name
        solver2: Second solver name
    """
    if solver1 not in results or solver2 not in results:
        print(f"Error: One solver missing from results")
        return

    # Only compare sizes that both solvers ran on
    sizes = [size for size in sizes_with_data(results, solver1) if results[solver2][size]]
    if not sizes:
        return

    plt.figure(figsize=(10, 6))

    percentage_diffs = []
    lower_errors = []
    upper_errors = []

    for size in sizes:
        log_ratios = np.log(np.maximum(results[solver2][size], 1e-9) / np.maximum(results[solver1][size], 1e-9))
        geo_mean = (np.exp(np.mean(log_ratios)) - 1) * 100
        p25, p75 = (np.exp(np.percentile(log_ratios, [25, 75])) - 1) * 100
        percentage_diffs.append(geo_mean)
        lower_errors.append(max(geo_mean - p25, 0))
        upper_errors.append(max(p75 - geo_mean, 0))

    # Create bar chart
    plt.bar(sizes, percentage_diffs, yerr=[lower_errors, upper_errors], capsize=3, alpha=0.7)

    # Baseline 0 horizontal line
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)

    plt.xlabel("Puzzle Size (n × n)")
    plt.ylabel(f"Time Difference: {solver2} / {solver1} - 1 (%)")
    plt.title(f"Runtime Difference (geometric mean): {solver2} vs {solver1}")
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    # Output directory
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/solver_difference_{solver1}_vs_{solver2}.png")
    plt.close()

def sizes_with_data(results, solver_name):
    """Sizes that a solver was actually run on (solvers can be capped with max_sizes)"""
    return [size for size in results["sizes"] if results[solver_name][size]]

def parse_sizes(value):
    """Parse a size argument like '5-15' or '5,7,9' into a list of ints"""
    if "-" in value:
        start, end = value.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(size) for size in value.split(",")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Queens solvers on the puzzles in /puzzles.")
    parser.add_argument("--solvers", nargs="+", default=list(ALL_SOLVERS.keys()),
                        choices=list(ALL_SOLVERS.keys()), help="Solvers to benchmark (quote names with spaces)")
    parser.add_argument("--sizes", type=parse_sizes, default=list(range(5, 16)), help="Sizes to run, e.g. '5-15' or '5,7,9'")
    parser.add_argument("--limit", type=int, default=100, help="Max puzzles per size (default 100)")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per solver per puzzle; the fastest is kept (default 3)")
    parser.add_argument("--backtrack-max-size", type=int, default=7,
                        help="Largest size to run BacktrackSolver on; it takes seconds per puzzle at 8x8 (default 7)")
    parser.add_argument("--row-backtrack-max-size", type=int, default=12,
                        help="Largest size to run RowBacktrackSolver on; it takes seconds per puzzle at 14x14 (default 12)")
    args = parser.parse_args()

    solvers = {name: ALL_SOLVERS[name] for name in args.solvers}
    max_sizes = {"BacktrackSolver": args.backtrack_max_size, "RowBacktrackSolver": args.row_backtrack_max_size}

    results = evaluate_solvers(solvers, sizes=args.sizes, limit=args.limit, repeats=args.repeats, max_sizes=max_sizes)
    print_summary(results)

    # Create visualizations
    plot_execution_time_vs_size(results, log_scale=False)
    plot_execution_time_vs_size(results, log_scale=True)
    solver_names = list(solvers.keys())
    for i, solver1 in enumerate(solver_names):
        for solver2 in solver_names[i+1:]:
            plot_execution_time_percentage_difference(results, solver1, solver2)
    print("Evaluation done.")
