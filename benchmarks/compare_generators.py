import argparse
import contextlib
import io
import random
import statistics
import time
from generators import generate_puzzle
from generators import solution_first
from benchmarks.evaluate_runtime import parse_sizes

GENERATORS = {
    "Random walk": lambda n: generate_puzzle.generate_puzzle(n, max_changes=1000),
    "Solution first": lambda n: solution_first.generate_puzzle(n),
}

def median_time(generate, n, runs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            generate(n)
        times.append(time.perf_counter() - start)
    return statistics.median(times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare median time to generate a unique puzzle.")
    parser.add_argument("--sizes", type=parse_sizes, default=[6, 8, 10, 12], help="Sizes to run, e.g. '6-10' or '6,8,10'")
    parser.add_argument("--runs", type=int, default=5, help="Puzzles per size per generator (default 5)")
    args = parser.parse_args()

    random.seed(0)
    print(f"{'Size':<6}" + "".join(f"{name:>16}" for name in GENERATORS))
    for n in args.sizes:
        times = [median_time(generate, n, args.runs) for generate in GENERATORS.values()]
        print(f"{n:<6}" + "".join(f"{t * 1000:>14.0f}ms" for t in times), flush=True)
