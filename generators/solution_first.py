import random
from solvers.pycosat_solver import PycosatSolver
from generators.generate_puzzle import number_of_islands, print_solution, visualize_board

def assigned_neighbour_regions(board, row, col):
    n = len(board)
    cells = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [board[r][c] for r, c in cells if 0 <= r < n and 0 <= c < n and board[r][c] != -1]

def random_queen_layout(n):
    """Random queen column for each row, with no two queens in the same column or touching"""
    def extend(cols):
        if len(cols) == n:
            return cols
        options = [c for c in range(n) if c not in cols and (not cols or abs(c - cols[-1]) > 1)]
        random.shuffle(options)
        for col in options:
            result = extend(cols + [col])
            if result:
                return result
        return None

    cols = extend([])
    if cols is None:
        raise ValueError(f"No valid queen layout exists for n={n}")
    return cols

def grow_regions(queen_cols):
    """Seed one region at each queen, then grow the regions with a random flood fill"""
    n = len(queen_cols)
    board = [[-1] * n for _ in range(n)]
    for row, region in zip(range(n), random.sample(range(n), n)):
        board[row][queen_cols[row]] = region

    unassigned = {(r, c) for r in range(n) for c in range(n) if board[r][c] == -1}
    while unassigned:
        frontier = [cell for cell in unassigned if assigned_neighbour_regions(board, *cell)]
        row, col = random.choice(frontier)
        board[row][col] = random.choice(assigned_neighbour_regions(board, row, col))
        unassigned.remove((row, col))
    return board

def move_cell(board, cells):
    """Move the first cell that can join a neighbouring region without disconnecting any region"""
    n = len(board)
    for row, col in cells:
        current = board[row][col]
        for region in set(assigned_neighbour_regions(board, row, col)) - {current}:
            board[row][col] = region
            if number_of_islands(board) == n:
                return True
            board[row][col] = current
    return False

def break_solution(board, solution, other):
    """
    Invalidate `other` by moving one of its queen cells into a neighbouring region, which then holds
    two of its queens. If no such move keeps regions connected, make a random move instead.
    Queens of `solution` never move, so `solution` stays valid and no region empties.
    """
    n = len(board)
    targeted = [(r, c) for r in range(n) for c in range(n) if other[r][c] and not solution[r][c]]
    fallback = [(r, c) for r in range(n) for c in range(n) if not solution[r][c]]
    random.shuffle(targeted)
    random.shuffle(fallback)
    return move_cell(board, targeted) or move_cell(board, fallback)

def generate_puzzle(n=9, max_moves=1000):
    """Generate a puzzle with exactly one solution. Returns the board and the number of moves used."""
    while True:
        queen_cols = random_queen_layout(n)
        solution = [[int(queen_cols[r] == c) for c in range(n)] for r in range(n)]
        board = grow_regions(queen_cols)
        # The solver holds a reference to board, so it sees each move without being rebuilt
        solver = PycosatSolver(board)
        for moves in range(max_moves):
            other = solver.find_other_solution(solution)
            if other is None:
                return board, moves
            if not break_solution(board, solution, other):
                break

if __name__ == "__main__":
    board, moves = generate_puzzle(n=8)
    print(f"Puzzle generated in {moves} moves")
    print_solution(board)
    visualize_board(board)
