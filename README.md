# Queens Solver

A LinkedIn-style Queens puzzle solver and generator using multiple constraint satisfaction approaches, including PycoSAT and Google's CP-SAT. This project implements efficient algorithms to solve the Queens puzzle and generate new puzzles.

## Project Overview

The Queens puzzle is a variant of the classic N-Queens problem with additional constraints. Each puzzle consists of:

1. An NxN grid divided into N distinct regions (colored areas)
2. The goal is to place N queens on the board such that:
   - Each row contains exactly one queen
   - Each column contains exactly one queen
   - Each region contains exactly one queen
   - No queen can diagonally attack another queen (no queens in adjacent diagonals)

This project provides several implementations to solve Queens puzzles:
- PycoSAT solver: Uses the PycoSAT boolean satisfiability solver
- CP-SAT solver: Uses Google's OR-Tools CP-SAT constraint programming solver
- Backtracking solver: A traditional backtracking algorithm solution

Additionally, the project includes puzzle generation capabilities and performance evaluation tools to compare different solving approaches.

## File Structure

- **Solvers:**
  - `pycosat_solver.py`: Implementation of a Queens puzzle solver using the PycoSAT library
  - `cpsat_solver.py`: Implementation using Google's OR-Tools CP-SAT solver
  - `backtrack_solver.py`: Implementation using traditional backtracking algorithm

- **Puzzle Generation:**
  - `generate_puzzle.py`: Functions to generate valid Queens puzzles of various sizes
  - `puzzles/`: Directory containing pre-generated puzzles of different sizes (5x5 to 16x16)

- **Evaluation:**
  - `evaluate_runtime.py`: Benchmarking code to compare the performance of different solvers
  - `figs/`: Directory containing performance comparison charts

- **Examples:**
  - `examples.py`: Sample puzzle boards for testing

- **Dependencies:**
  - `requirements.txt`: Required Python packages

## How It Works

### Solvers

1. **PycoSAT Solver:**
   - Encodes the Queens puzzle constraints as boolean satisfiability (SAT) clauses
   - Implements both basic constraints and advanced heuristics to improve solving efficiency

2. **CP-SAT Solver:**
   - Uses constraint programming with Google's OR-Tools
   - Implements both basic and advanced heuristics for efficient puzzle solving

3. **Backtracking Solver:**
   - Implements a traditional recursive backtracking algorithm
   - Systematically tries different queen placements and backs up when constraints are violated

### Puzzle Generation

The puzzle generator creates valid Queens puzzles by:
1. Creating an initial board with N connected regions
2. Iteratively modifying the board while maintaining region connectivity
3. Ensuring the puzzle has exactly one solution

### Performance Evaluation

The evaluation module benchmarks different solvers across various puzzle sizes and generates comparative visualizations.

## Installation and Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/queens-solver.git
   cd queens-solver
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

### Solving a Puzzle

You can solve a pre-defined Queens puzzle using any of the three solvers:

```python
from pycosat_solver import PycosatSolver
from cpsat_solver import CpSatSolver
from backtrack_solver import BacktrackSolver
from examples import create_test_boards

# Load a sample board
board = create_test_boards()[0]

# Solve using PycoSAT
pycosat_solver = PycosatSolver(board)
solution = pycosat_solver.solve()
pycosat_solver.print_solution()

# Or solve using CP-SAT
cpsat_solver = CpSatSolver(board)
solution = cpsat_solver.solve()
cpsat_solver.print_solution()

# Or solve using backtracking
backtrack_solver = BacktrackSolver(board)
solution = backtrack_solver.solve()
backtrack_solver.print_solution()
```

### Generating Puzzles

To generate a new puzzle:

```python
from generate_puzzle import generate_puzzle, visualize_board

# Generate a 9x9 puzzle
board = generate_puzzle(n=9)

# Visualize the generated puzzle
visualize_board(board)
```

To generate multiple puzzles and save them:

```python
from generate_puzzle import generate_multiple_puzzles

# Generate 10 puzzles of size 9x9
generate_multiple_puzzles(n=9, num_puzzles=10, output_dir="puzzles")
```

### Evaluating Solver Performance

To compare the performance of different solvers:

```python
from evaluate_runtime import evaluate_solvers, plot_execution_time_vs_size
from pycosat_solver import PycosatSolver
from cpsat_solver import CpSatSolver
from backtrack_solver import BacktrackSolver

solvers = {
    "PycosatSolver": PycosatSolver,
    "PycosatSolver (with heuristics)": lambda board: PycosatSolver(board, use_heuristic=True),
    "CpSatSolver": CpSatSolver,
    "CpSatSolver (with heuristics)": lambda board: CpSatSolver(board, use_heuristic=True)
}

results = evaluate_solvers(solvers, sizes=[5, 6, 7, 8, 9])
plot_execution_time_vs_size(results)
```

## Testing

The project includes example puzzles with known solutions in `examples.py` that can be used for testing. Each solver implementation also includes its own test cases.

To run a simple test with all solvers:

```python
from examples import create_test_boards
from pycosat_solver import PycosatSolver
from cpsat_solver import CpSatSolver
from backtrack_solver import BacktrackSolver

test_boards = create_test_boards()
for board in test_boards:
    # Test PycoSAT solver
    solver1 = PycosatSolver(board)
    solution1 = solver1.solve()
    
    # Test CP-SAT solver
    solver2 = CpSatSolver(board)
    solution2 = solver2.solve()
    
    # Test backtracking solver
    solver3 = BacktrackSolver(board)
    solution3 = solver3.solve()
    
    # Verify solutions match
    assert solution1 == solution2 == solution3
```

## Performance Results

Our evaluation shows that:
1. CP-SAT solver generally outperforms PycoSAT on larger puzzles
2. Both solvers significantly outperform traditional backtracking
3. The heuristic optimizations substantially improve performance for both SAT solvers

Detailed performance comparisons can be found in the `figs/` directory.

## Dependencies

- Python 3.6+
- pycosat: For SAT solving
- ortools: For CP-SAT constraint programming
- numpy: For numerical operations
- matplotlib: For visualization

## Contributors

- Wesley Liu
- Sean Fang