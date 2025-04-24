import random
from pycosat_solver import PycosatSolver
from backtrack_solver import BacktrackSolver
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

def generate_puzzle(n=9, max_changes=1000):
    """
    Randomly generate a Queens puzzle with the given size.
    
    Args:
        size: the size of the board
        max_changes: the maximum number of changes to make to the board
    """
    board = create_start_board(n)

    # Solve the puzzle and change cells until we get a valid board with one solution
    solver = PycosatSolver(board, use_heuristic=True)
    changes_count = 0
    while not solver.has_one_solution():
        # Randomly iterate through cells until we find a boundary cell, then flip the region assignment
        all_cells = [(row, col) for row in range(n) for col in range(n)]
        random.shuffle(all_cells)

        found_boundary = False
        for row, col in all_cells:
            adjacent_cells = [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]
            valid_adjacent_cells = [(c_row, c_col) for c_row, c_col in adjacent_cells if 0 <= c_row < n and 0 <= c_col < n]
            assigned_neighbor_regions = [board[c_row][c_col] for c_row, c_col in valid_adjacent_cells if board[c_row][c_col] != -1]
            # shuffle assigned neighbor regions so we don't favor bottom left
            random.shuffle(assigned_neighbor_regions)
            for region in assigned_neighbor_regions:
                if region != board[row][col]:
                    prev = board[row][col]
                    board[row][col] = region
                    # Check that all regions are still connected, and if not, revert change
                    if number_of_islands(board) != n:
                        board[row][col] = prev
                    else:
                        found_boundary = True
                        break
            if found_boundary:
                break

        changes_count += 1

        # Reroll the start board if we've been trying for too long, or if no boundary changes are possible
        if not found_boundary or changes_count > max_changes:
            board = create_start_board(n)
            changes_count = 0
            print("Rerolling start board")

        solver = PycosatSolver(board, use_heuristic=True)
    print(f"Puzzle generated in {changes_count} changes")
    return board

def create_start_board(n=9):
    """
    Create a start board with n regions. This populates the board with n regions but doesn't ensure that there is only one solution.
    """
    # -1 means unassigned, 0-(n-1) means region assignment
    board = [[-1 for _ in range(n)] for _ in range(n)]

    boundaries = set()

    # Randomly assign n starting blocks for regions
    starting_regions_assigned = 0
    while starting_regions_assigned < n:
        # Choose a random cell and try to assign a new region
        row = random.randint(0, n-1)
        col = random.randint(0, n-1)
        if board[row][col] == -1:
            board[row][col] = starting_regions_assigned
            starting_regions_assigned += 1
            # Add adjacent cells to boundaries to be explored later
            adjacent_cells = [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]
            for c_row, c_col in adjacent_cells:
                if 0 <= c_row < n and 0 <= c_col < n and board[c_row][c_col] == -1:
                    boundaries.add((c_row, c_col))
    
    # Assign regions to the rest of the board, kind of like BFS or flood fill
    while boundaries:
        # Choose a random boundary cell
        row, col = random.choice(list(boundaries))
        boundaries.remove((row, col))

        # Assign a region to the cell based on neighboring cells (at least one must be assigned)
        adjacent_cells = [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]
        valid_adjacent_cells = [(c_row, c_col) for c_row, c_col in adjacent_cells if 0 <= c_row < n and 0 <= c_col < n]
        assigned_neighbor_regions = [board[c_row][c_col] for c_row, c_col in valid_adjacent_cells if board[c_row][c_col] != -1]
        if not assigned_neighbor_regions:
            continue
        board[row][col] = random.choice(assigned_neighbor_regions)

        # Add adjacent cells to boundaries to be explored later
        unassigned_neighbors = [(c_row, c_col) for c_row, c_col in valid_adjacent_cells if board[c_row][c_col] == -1]
        boundaries.update(unassigned_neighbors)

    # Check that all cells are assigned a region
    for row in range(n):
        for col in range(n):
            if board[row][col] == -1:
                raise ValueError("Not all cells were assigned a region")
            
    return board

def number_of_islands(board):
    """
    Count the number of islands in the board with BFS. Used to make sure colors are connected.
    """
    n = len(board)
    count = 0
    visited = [[False for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for col in range(n):
            if not visited[row][col]:
                count += 1
                region = board[row][col]
                visited[row][col] = True
                stack = [(row, col)]
                while stack:
                    r, c = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == region and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
    return count

def print_solution(board):
    """Pretty print the board"""
    if board is None:
        print("No board.")
        return
    
    for row in board:
        print(' '.join(str(cell) for cell in row))

def visualize_board(board):
    """
    Display a visual representation of the board with different colors for each region.
    
    Args:
        board: 2D array representing the puzzle board
        show_numbers: Whether to display region numbers on cells
    """
    if board is None:
        print("No board to visualize.")
        return
    
    # Convert board to numpy array
    board_array = np.array(board)
    n = len(board)
    
    # Create a distinct color map
    num_regions = n 
    
    # Generate distinct colors using HSV color space for better visual distinction
    colors = []
    for i in range(num_regions):
        hue = i / num_regions
        # Avoid low saturation or value which could result in colors too similar to white/black
        saturation = 0.7 + random.uniform(0, 0.3)
        value = 0.7 + random.uniform(0, 0.3)
        colors.append(mcolors.hsv_to_rgb([hue, saturation, value]))
    
    # Create a custom colormap
    cmap = mcolors.ListedColormap(colors)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(board_array, cmap=cmap, vmin=0, vmax=num_regions-1)
    
    # Add grid lines
    for i in range(n+1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title('Puzzle Board')
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def convert_board_to_txt_line(board):
    """
    Convert board to line of text with comma-separated values, handles any size board (double digits).
    """
    return ','.join(str(cell) for row in board for cell in row)

def convert_txt_line_to_board(line):
    """
    Convert a comma-separated text line to a board with int values.
    """
    values = line.split(',')
    n = int(len(values) ** 0.5)
    return [[int(values[i * n + j]) for j in range(n)] for i in range(n)]

def generate_multiple_puzzles(n=9, num_puzzles=100, max_changes=10000, output_dir="puzzles"):
    """
    Generate multiple puzzles of size nxn and save them to a file in a directory. Note that this can technically
    lead to duplicate puzzles, but the probability is low.
    """
    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save puzzles
    puzzles = [generate_puzzle(n, max_changes) for _ in range(num_puzzles)]
    with open(f'{output_dir}/size_{n}.txt', 'a') as f:
        for i, puzzle in enumerate(puzzles):
            f.write(convert_board_to_txt_line(puzzle) + '\n')

    print(f"Generated {num_puzzles} puzzles of size {n}x{n}")

if __name__ == "__main__":
    # for n in range(16, 18):
    #     generate_multiple_puzzles(n, 100, 500)

    # Generate a puzzle
    board = generate_puzzle(n=9, max_changes=1000)
    assert board == convert_txt_line_to_board(convert_board_to_txt_line(board))
    print_solution(board)
    
    # Verify using the solver
    print("\nVerifying with PycosatSolver:")
    pycosat_solver = PycosatSolver(board)
    pycosat_solution = pycosat_solver.solve()
    if pycosat_solution:
        pycosat_solver.print_solution()
    else:
        print("No solution found by PycosatSolver. The puzzle may be invalid.")
    print("\nVerifying with BacktrackSolver:")
    # backtrack_solver = BacktrackSolver(board)
    # backtrack_solution = backtrack_solver.solve()
    # if backtrack_solution:
    #     backtrack_solver.print_solution()
    # else:
    #     print("No solution found by BacktrackSolver. The puzzle may be invalid.")
    # assert(pycosat_solution == backtrack_solution)

    # Visualize the board
    visualize_board(board)
