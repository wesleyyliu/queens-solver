# Queens Solver

A LinkedIn-style Queens puzzle solver and generator using multiple constraint satisfaction approaches, including PycoSAT and Google's CP-SAT. This project implements efficient algorithms to solve the Queens puzzle and generate new puzzles, utilizing heuristics.

## Project Overview

The Queens puzzle is a variant of the classic N-Queens problem with additional constraints. Each puzzle consists of:

1. An NxN grid divided into N distinct regions (colored areas)
2. The goal is to place N queens on the board such that:
   - Each row contains exactly one queen
   - Each column contains exactly one queen
   - Each region contains exactly one queen
   - No queen are placed diagonally adjacent to another queen

This project provides several implementations to solve Queens puzzles:
- PycoSAT solver: Uses the PycoSAT boolean satisfiability solver
- CP-SAT solver: Uses Google's OR-Tools CP-SAT constraint programming solver
- Backtracking solver: A traditional backtracking algorithm solution

Additionally, the project includes puzzle generation capabilities and performance evaluation tools to compare different solving approaches, while simultaneously testing the accuracy of different solvers.

## File Structure

- **Solvers (`solvers/`):**
  - `base.py`: `QueensSolver` base class (input validation, shared interface) and `is_valid_solution`
  - `pycosat_solver.py`: Implementation of a Queens puzzle solver using the PycoSAT library
  - `cpsat_solver.py`: Implementation using Google's OR-Tools CP-SAT solver
  - `backtrack_solver.py`: Naive backtracking, plus `RowBacktrackSolver` (one queen per row)

- **Puzzle Generation (`generators/`):**
  - `generate_puzzle.py`: Random regions, then random border moves until the solution is unique
  - `solution_first.py`: Places a valid queen layout first, then breaks each second solution found
  - `puzzles/`: Directory containing pre-generated puzzles of different sizes (5x5 to 16x16)

- **Evaluation (`benchmarks/`):**
  - `evaluate_runtime.py`: Benchmarking code to compare the performance of different solvers
  - `compare_generators.py`: Time to generate a unique puzzle with each generator
  - `figs/`: Directory containing performance comparison charts saved by evaluate_runtime.py

- **Tests (`tests/`):**
  - `test_queens.py`: Solver correctness, heuristic soundness and generator checks (run `pytest`)

- **Initial Example:**
  - `examples.py`: Sample puzzle board that was initially used manually for testing.

- **Dependencies:**
  - `requirements.txt`: Required Python packages

## How It Works

### Solvers

1. **PycoSAT Solver (`pycosat_solver.py`):**
   - Encodes the Queens puzzle constraints as boolean satisfiability (SAT) clauses and solves it
   - `add_queen_constraints()`: Adds clauses ensuring exactly one queen per row and one queen per column
   - `add_region_constraints()`: Adds clauses ensuring exactly one queen per region
   - `add_diagonal_adjacency_constraints()`: Prevents queens from being placed diagonally adjacent to each other
   - `add_heuristic_constraints()`: Implements heuristics to improve solving efficiency, including:
     - Automatically placing queens in singleton regions (regions with only one cell)
     - Eliminating cells of different regions when a region occupies only a single row or column
     - Eliminating cells of different regions when two regions occupies only the exact same two rows or columns
   - `solve()`: Combines all constraints, invokes the PycoSAT solver, and extracts the solution
   - `has_one_solution()`: Verifies that the puzzle has exactly one solution

2. **CP-SAT Solver (`cpsat_solver.py`):**
   - Uses Google's OR-Tools CP-SAT constraint programming solver
   - `add_row_constraints()`: Ensures exactly one queen per row
   - `add_column_constraints()`: Ensures exactly one queen per column
   - `add_vicinity_constraints()`: Prevents queens from being diagonally adjacent
   - `add_region_constraints()`: Ensures exactly one queen per region
   - `add_heuristic_constraints()`: Implements advanced solving heuristics:
     - Identifies singleton regions and places queens automatically
     - Applies row/column exclusions for regions constrained to specific rows/columns
   - `add_pairwise_region_column_row_heuristic()`: Eliminates cells of different regions when two regions occupies only the exact same two rows or columns
   - `get_possible_cells_for_hinting()`: Computes probable queen placements for hinting mid-solve
   - `solve()`: Sets up the constraint model, solves it, and extracts the solution

3. **Backtracking Solver (`backtrack_solver.py`):**
   - Implements a traditional recursive backtracking algorithm
   - `is_safe()`: Checks if a queen can be placed at a given position without violating constraints
   - `backtrack()`: Recursively tries different queen placements and backs up when constraints are violated
   - `solve()`: Coordinates the backtracking process and returns the solution

### Puzzle Generation (`generate_puzzle.py`)

The puzzle generator creates valid Queens puzzles through the following functions:

- `generate_puzzle()`: Main function that creates valid puzzles with exactly one solution by:
  1. Creating an initial board with N connected regions
  2. Iteratively modifying the board while maintaining region connectivity
  3. Testing solutions until a board with exactly one solution is found

- `create_start_board()`: Creates an initial board with N connected regions using a randomized approach:
  1. Assigns N starting cells to represent each region
  2. Expands regions in a flood-fill manner to cover the entire board

- `number_of_islands()`: Validates that regions remain connected during modifications

- `visualize_board()`: Creates a colored visualization of the puzzle using matplotlib

- `convert_board_to_txt_line()`: Converts a board to a compact text representation for storage

- `convert_txt_line_to_board()`: Converts stored text back to a board representation

- `generate_multiple_puzzles()`: Generates and saves multiple puzzle instances of a specified size to /puzzles

### Performance Evaluation (`evaluate_runtime.py`)

The evaluation file benchmarks different solvers with the following functions:

- `evaluate_solvers()`: Benchmarks specified solvers on puzzles of different sizes:
  1. Loads puzzles from the puzzles directory
  2. Times each solver on each puzzle
  3. Verifies that all solvers produce the same solution
  4. Collects and returns timing statistics

- `plot_execution_time_vs_size()`: Generates comparison charts with error bars showing:
  1. Average execution time for each solver across different puzzle sizes
  2. Standard deviation of execution times

- `plot_execution_time_percentage_difference()`: Creates visualizations comparing the relative performance of solver pairs

## Installation and Setup

1. Clone this repository:
   ```
   git clone https://github.com/wesleyyliu/queens-solver.git
   cd queens-solver
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Quick Demo

Run from the repository root with the virtual environment active:

```
python -m solvers.pycosat_solver          # solve the sample puzzle
python -m generators.solution_first       # generate a unique 8x8 puzzle and show it
pytest                                    # run the tests
python -m benchmarks.evaluate_runtime --solvers PycosatSolver CpSatSolver RowBacktrackSolver --limit 20
```

The benchmark prints a summary table and saves charts to `figs/` (`execution_time_vs_size_log.png` is the main one).

## Running the Project

### Solving Puzzles:

Run commands from the repository root. Run any solver module directly to test a sample puzzle:

```
python -m solvers.backtrack_solver
python -m solvers.pycosat_solver
python -m solvers.cpsat_solver
```

### Generating Puzzles:

To generate a puzzle and visualize the output, run either generator:
```
python -m generators.generate_puzzle
python -m generators.solution_first
```

To generate multiple puzzles and save the output to puzzles/, put 
```
generate_multiple_puzzles(n, 100, 500)
```
in generators/generate_puzzle.py and run it (with your desired choice of n).

### Evaluation

```
python -m benchmarks.evaluate_runtime                      # all solvers; see --help for options
python -m benchmarks.compare_generators --sizes 6,8,10,12
pytest
```

## Dependencies

All dependencies are outlined in `requirements.txt`. The Python version must be 3.6+.

## Contributors

- Wesley Liu
- Sean Fang