from abc import ABC, abstractmethod

class QueensSolver(ABC):
    def __init__(self, board):
        n = len(board)
        if any(len(row) != n for row in board):
            raise ValueError("Board must be square")
        if {cell for row in board for cell in row} != set(range(n)):
            raise ValueError(f"Board must use region ids 0 to {n - 1}")
        self.board = board
        self.n = n
        self.solution = None

    @abstractmethod
    def solve(self):
        """Return an n x n grid with 1 where a queen goes, or None if there is no solution."""

    def print_solution(self):
        """Pretty print the final solution"""
        if self.solution is None:
            print("No solution exists.")
            return

        for row in self.solution:
            print(' '.join('Q' if cell == 1 else '.' for cell in row))

def is_valid_solution(board, solution):
    """Check a solution against the rules: one queen per row, column and region, and no queens touching."""
    n = len(board)
    if solution is None or len(solution) != n:
        return False
    queens = {(row, col) for row in range(n) for col in range(n) if solution[row][col] == 1}
    return (len(queens) == n
            and len({row for row, _ in queens}) == n
            and len({col for _, col in queens}) == n
            and len({board[row][col] for row, col in queens}) == n
            and not any((row + 1, col + d) in queens for row, col in queens for d in (-1, 1)))
