import random
import pycosat
import pytest
from solvers.backtrack_solver import BacktrackSolver, RowBacktrackSolver
from solvers.cpsat_solver import CpSatSolver
from examples import create_test_boards
from generators.generate_puzzle import convert_txt_line_to_board, create_start_board, number_of_islands
from solvers.pycosat_solver import PycosatSolver
from generators import solution_first
from solvers.base import is_valid_solution

SOLVERS = [
    PycosatSolver,
    lambda board: PycosatSolver(board, use_heuristic=True),
    CpSatSolver,
    lambda board: CpSatSolver(board, use_heuristic=True),
    BacktrackSolver,
    RowBacktrackSolver,
]

def load_puzzles(n, count):
    with open(f"puzzles/size_{n}.txt") as f:
        return [convert_txt_line_to_board(line) for line in f.readlines()[:count]]

def all_solutions(board, use_heuristic):
    clauses = PycosatSolver(board, use_heuristic=use_heuristic).build_clauses()
    return {frozenset(v for v in result if v > 0) for result in pycosat.itersolve(clauses)}

@pytest.mark.parametrize("make_solver", SOLVERS)
def test_example_board(make_solver):
    board = create_test_boards()[0]
    expected = [2, 6, 3, 5, 1, 4, 0]
    solution = make_solver(board).solve()
    assert [row.index(1) for row in solution] == expected

@pytest.mark.parametrize("make_solver", SOLVERS)
@pytest.mark.parametrize("n", [5, 6, 7])
def test_dataset_puzzles(make_solver, n):
    for board in load_puzzles(n, 5):
        assert is_valid_solution(board, make_solver(board).solve())

@pytest.mark.parametrize("make_solver", SOLVERS)
def test_unsolvable_board_returns_none(make_solver):
    assert make_solver([[0, 0], [1, 1]]).solve() is None

@pytest.mark.parametrize("board", [[[0, 1]], [[0, 0], [0, 0]], [[0, 1], [1, 2]]])
def test_invalid_board_raises(board):
    with pytest.raises(ValueError):
        PycosatSolver(board)

@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_heuristics_keep_every_solution(n):
    random.seed(n)
    for _ in range(25):
        board = create_start_board(n)
        assert all_solutions(board, use_heuristic=True) == all_solutions(board, use_heuristic=False)

@pytest.mark.parametrize("n", [5, 6, 8, 10])
def test_solution_first_generator(n):
    random.seed(n)
    board, _ = solution_first.generate_puzzle(n)
    assert number_of_islands(board) == n
    assert PycosatSolver(board).has_one_solution()
