import pycosat
from examples import create_test_boards

class PycosatSolver:
    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.clauses = []
        self.solution = None

    def var_index(self, row, col):
        """Convert row, col position to corresponding variable index (1-based in pycosat)"""
        return row * self.n + col + 1
    
    def row_col(self, var_index):
        """Convert variable index to row, col position"""
        return (var_index - 1) // self.n, (var_index - 1) % self.n
    
    def add_queen_constraints(self):
        """Add constraints for n queens being placed"""
        
        # Row constraints for 1 queen per row
        for row in range(self.n):
            # >= 1 queen per row
            self.clauses.append([self.var_index(row, col) for col in range(self.n)])
            
            # <= 1 queen per row
            for col1 in range(self.n):
                for col2 in range(col1 + 1, self.n):
                    self.clauses.append([-self.var_index(row, col1), -self.var_index(row, col2)])
        
        # Column constraints for 1 queen per column
        for col in range(self.n):
            # >= 1 queen per column
            self.clauses.append([self.var_index(row, col) for row in range(self.n)])
            
            # <= 1 queen per column
            for row1 in range(self.n):
                for row2 in range(row1 + 1, self.n):
                    self.clauses.append([-self.var_index(row1, col), -self.var_index(row2, col)])
        
    def add_region_constraints(self):
        """Add constraints for one queen per region"""
        
        # Number of regions should be equal to number of queens, and 0-indexed
        regions = [i for i in range(self.n)]

        # This for loop ensures each region has 1 queen
        for region in regions:
            positions = []
            for row in range(self.n):
                for col in range(self.n):
                    if self.board[row][col] == region:
                        positions.append((row, col))
            
            # >= 1 queen in region
            self.clauses.append([self.var_index(row, col) for row, col in positions])
            
            # <= 1 queen in region
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    self.clauses.append([-self.var_index(x1, y1), -self.var_index(x2, y2)])
    
    def add_diagonal_adjacency_constraints(self):
        """Add constraints to prevent queens from being diagonally adjacent"""
        
        for row in range(self.n):
            for col in range(self.n):
                # Check diagonal adjacency
                diagonals = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
                
                for d_row, d_col in diagonals:
                    if 0 <= d_row < self.n and 0 <= d_col < self.n:
                        # <= 1 queen in diagonal
                        self.clauses.append([-self.var_index(row, col), -self.var_index(d_row, d_col)])
    
    def solve(self):
        """Set up constraints and solve queens yay"""
        
        print("Solving queens puzzle with pycosat...")

        self.add_queen_constraints()
        self.add_region_constraints()
        self.add_diagonal_adjacency_constraints()
        
        # All true literals returned
        result = pycosat.solve(self.clauses)
        if result == "UNSAT":
            raise ValueError('Impossible Queens puzzle!')
        
        self.solution = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for var in result:
            if var > 0:
                row, col = self.row_col(var)
                self.solution[row][col] = 1
        
        return self.solution
    
    def print_solution(self):
        """Pretty print the final solution"""
        if self.solution is None:
            print("No solution exists.")
            return
        
        for row in self.solution:
            print(' '.join('Q' if cell == 1 else '.' for cell in row))

if __name__ == "__main__":
    test_boards = create_test_boards()
    for board in test_boards:
        solver = PycosatSolver(board)
        solution = solver.solve()
        solver.print_solution()