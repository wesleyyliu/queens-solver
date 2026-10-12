import pycosat
from examples import create_test_boards
from solvers.base import QueensSolver

class PycosatSolver(QueensSolver):
    def __init__(self, board, use_heuristic=False):
        super().__init__(board)
        self.clauses = []
        self.static_clauses = None
        self.use_heuristic = use_heuristic

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
        """Add constraints to prevent queens from being diagonally adjacent (each pair once)"""

        for row in range(self.n - 1):
            for col in range(self.n):
                for d_col in (col - 1, col + 1):
                    if 0 <= d_col < self.n:
                        self.clauses.append([-self.var_index(row, col), -self.var_index(row + 1, d_col)])

    def add_heuristic_constraints(self):
        """Add extra heuristic constraints to possibly speed up solver"""

        region_positions = {}
        regions_rows = [set() for _ in range(self.n)]
        regions_cols = [set() for _ in range(self.n)]

        for row in range(self.n):
            for col in range(self.n):
                region = self.board[row][col]
                if region not in region_positions:
                    region_positions[region] = []
                region_positions[region].append((row, col))
                regions_rows[region].add(row)
                regions_cols[region].add(col)

        # If there is only one position for a region, just make it a queen
        for region in region_positions:
            positions = region_positions[region]
            if len(positions) == 1:
                row, col = positions[0]
                self.clauses.append([self.var_index(row, col)])

        for region in range(self.n):
            if len(regions_rows[region]) == 1:
                # should just be one row
                for row in regions_rows[region]:
                    for col in range(self.n):
                        # For cells in this row not in the same region, add negative constraint
                        if self.board[row][col] != region:
                            self.clauses.append([-self.var_index(row, col)])
            if len(regions_cols[region]) == 1:
                # should just be one col
                for col in regions_cols[region]:
                    for row in range(self.n):
                        # For cells in this col not in the same region, add negative constraint
                        if self.board[row][col] != region:
                            self.clauses.append([-self.var_index(row, col)])


        for r1 in range(self.n):
            for r2 in range(r1 + 1, self.n):
                s1 = regions_rows[r1]
                s2 = regions_rows[r2]
                if len(s1) == 2 and len(s2) == 2 and s1 == s2:
                    # Add negative constraints for all cells in these two rows that are not in r1 or r2
                    for row in s1:
                        for col in range(self.n):
                            current_region = self.board[row][col]
                            if current_region != r1 and current_region != r2:
                                self.clauses.append([-self.var_index(row, col)])

                s1 = regions_cols[r1]
                s2 = regions_cols[r2]
                if len(s1) == 2 and len(s2) == 2 and s1 == s2:
                    # Add negative constraints for all cells in these two cols that are not in r1 or r2
                    for col in s1:
                        for row in range(self.n):
                            current_region = self.board[row][col]
                            if current_region != r1 and current_region != r2:
                                self.clauses.append([-self.var_index(row, col)])

    def build_clauses(self):
        """Build the full CNF for this board. Row, column and diagonal clauses only depend on n, so they're cached."""
        if self.static_clauses is None:
            self.clauses = []
            self.add_queen_constraints()
            self.add_diagonal_adjacency_constraints()
            self.static_clauses = self.clauses
        self.clauses = list(self.static_clauses)
        self.add_region_constraints()
        if self.use_heuristic:
            self.add_heuristic_constraints()
        return self.clauses

    def decode(self, result):
        """Convert a pycosat assignment into a 0/1 grid"""
        solution = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for var in result:
            if var > 0:
                row, col = self.row_col(var)
                solution[row][col] = 1
        return solution

    def solve(self):
        """Set up constraints and solve queens yay"""
        result = pycosat.solve(self.build_clauses())
        self.solution = None if result == "UNSAT" else self.decode(result)
        return self.solution

    def find_other_solution(self, solution):
        """Return a solution different from the given one, or None if there isn't one"""
        # Every solution has exactly n queens, so "not all of these queens" excludes exactly this solution
        blocking_clause = [-self.var_index(row, col) for row in range(self.n) for col in range(self.n) if solution[row][col]]
        result = pycosat.solve(self.build_clauses() + [blocking_clause])
        return None if result == "UNSAT" else self.decode(result)

    def has_one_solution(self):
        """Return True if the queens puzzle has exactly one solution"""
        solutions = pycosat.itersolve(self.build_clauses())
        return next(solutions, None) is not None and next(solutions, None) is None

if __name__ == "__main__":
    test_boards = create_test_boards()
    for board in test_boards:
        solver = PycosatSolver(board)
        solution = solver.solve()
        solver.print_solution()
