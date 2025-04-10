from examples import create_test_boards

class BacktrackSolver:
    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.solution = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.regions_used = [False] * self.n
        self.rows_used = [False] * self.n
        self.cols_used = [False] * self.n

    def is_safe(self, row, col):
        """Check if it's safe to place a queen at position (row, col)"""
        # Check if this row or column already has a queen
        if self.rows_used[row] or self.cols_used[col]:
            return False
        
        # Check if the region already has a queen
        region = self.board[row][col]
        if self.regions_used[region]:
            return False
        
        # Check for queens in adjacent diagonal positions (within 1 block)
        diagonal_adjacents = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
        
        for adj_row, adj_col in diagonal_adjacents:
            # Check if position is valid and contains a queen
            if (0 <= adj_row < self.n and 0 <= adj_col < self.n and 
                self.solution[adj_row][adj_col] == 1):
                return False
        
        return True
    
    def backtrack(self, queens_placed):
        """Recursive backtracking function to place queens"""
        if queens_placed == self.n:
            return True
        
        # Try placing a queen in each cell
        for row in range(self.n):
            for col in range(self.n):
                region = self.board[row][col]
                
                if self.is_safe(row, col):
                    # Place the queen
                    self.solution[row][col] = 1
                    self.rows_used[row] = True
                    self.cols_used[col] = True
                    self.regions_used[region] = True
                    
                    # Recur to place the rest of the queens
                    if self.backtrack(queens_placed + 1):
                        return True
                    
                    # If placing a queen here doesn't lead to a solution, backtrack
                    self.solution[row][col] = 0
                    self.rows_used[row] = False
                    self.cols_used[col] = False
                    self.regions_used[region] = False
        
        return False
    
    def solve(self):
        """
        Solves the LinkedIn Queens puzzle using backtracking.
        
        Args:
            board: A nxn matrix where each cell contains a value 0-(n-1) representing its region.
            
        Returns:
            A solution matrix where 1 represents a queen and 0 represents an empty cell.
            None if no solution exists.
        """

        print("Solving the queens puzzle with backtracking...")
        if self.backtrack(0):
            return self.solution
        else:
            return None

    def print_solution(self):
        """Pretty print the final solution"""
        if self.solution is None:
            print("No solution exists.")
            return
        
        for row in self.solution:
            print(' '.join('Q' if cell == 1 else '.' for cell in row))

if __name__ == "__main__":
    test_boards = create_test_boards()
    for test_board in test_boards:
        solver = BacktrackSolver(test_board)
        solution = solver.solve()
        solver.print_solution()
    