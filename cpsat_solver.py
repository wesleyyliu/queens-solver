from ortools.sat.python import cp_model
from examples import create_test_boards

class CpSatSolver:
    def __init__(self, board, use_heuristic=False):
        """
        Initialize the solver with a board
        board: 2D list, board[i][j] gives region number for cell (i, j)
        use_heuristic:
        """
        self.board = board
        self.n = len(board)
        self.solution = None
        self.use_heuristic = use_heuristic

    def add_row_constraints(self, model, x):
        n = self.n
        for i in range(n):
            model.AddExactlyOne(x[i][j] for j in range(n))

    def add_column_constraints(self, model, x):
        n = self.n
        for j in range(n):
            model.AddExactlyOne(x[i][j] for i in range(n))

    def add_vicinity_constraints(self, model, x):
        """Add constraints so no two queens are in diagonally adjacent (corner) cells (no redundancy)."""
        n = self.n
        for i in range(n):
            for j in range(n):
                # Only check corners (diagonals)
                for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        # Only add if (i, j) < (ni, nj) lexicographically
                        if (i < ni) or (i == ni and j < nj):
                            model.Add(x[i][j] + x[ni][nj] <= 1)

    def add_region_constraints(self, model, x):
        region_map = {}
        n = self.n
        for i in range(n):
            for j in range(n):
                region = self.board[i][j]
                if region not in region_map:
                    region_map[region] = []
                region_map[region].append(x[i][j])
        for region_vars in region_map.values():
            model.AddAtMostOne(region_vars)

    def add_heuristic_constraints(self, model, x):
        """Add extra heuristic constraints to possibly speed up solver (region/row/col singletons and more)."""
        n = self.n
        region_positions = {}
        regions_rows = [set() for _ in range(n)]
        regions_cols = [set() for _ in range(n)]
        for row in range(n):
            for col in range(n):
                region = self.board[row][col]
                if region not in region_positions:
                    region_positions[region] = []
                region_positions[region].append((row, col))
                regions_rows[region].add(row)
                regions_cols[region].add(col)
        # Region singleton
        for region in region_positions:
            positions = region_positions[region]
            if len(positions) == 1:
                row, col = positions[0]
                model.Add(x[row][col] == 1)
                if (row > 0) and (col > 0):
                    model.Add(x[row - 1][col - 1] == 0)
                if (row > 0) and (col < n - 1):
                    model.Add(x[row - 1][col + 1] == 0)
                if (row < n - 1) and (col > 0):
                    model.Add(x[row + 1][col - 1] == 0)
                if (row < n - 1) and (col < n - 1):
                    model.Add(x[row + 1][col + 1] == 0)
                else:
                    pass
        # Row/col singleton for region
        for region in range(n):
            if len(regions_rows[region]) == 1:
                row = list(regions_rows[region])[0]
                for col in range(n):
                    if self.board[row][col] != region:
                        model.Add(x[row][col] == 0)
            if len(regions_cols[region]) == 1:
                col = list(regions_cols[region])[0]
                for row in range(n):
                    if self.board[row][col] != region:
                        model.Add(x[row][col] == 0)
        

    def add_pairwise_region_column_row_heuristic(self, model, x):
        """If two regions together only fill two columns or rows, eliminate all other regions from those columns/rows (X-Wing heuristic)."""
        n = self.n
        # Build region -> set of columns and rows mapping
        region_positions = {}
        for row in range(n):
            for col in range(n):
                region = self.board[row][col]
                if region not in region_positions:
                    region_positions[region] = []
                region_positions[region].append((row, col))
        region_to_cols = {region: set(col for (_, col) in cells) for region, cells in region_positions.items()}
        region_to_rows = {region: set(row for (row, _) in cells) for region, cells in region_positions.items()}
        region_list = list(region_positions.keys())
        # Columns version
        for i in range(len(region_list)):
            for j in range(i + 1, len(region_list)):
                r1, r2 = region_list[i], region_list[j]
                cols = region_to_cols[r1] | region_to_cols[r2]
                if len(cols) == 2:
                    for col in cols:
                        for row in range(n):
                            reg = self.board[row][col]
                            if reg != r1 and reg != r2:
                                model.Add(x[row][col] == 0)
        # Rows version
        for i in range(len(region_list)):
            for j in range(i + 1, len(region_list)):
                r1, r2 = region_list[i], region_list[j]
                rows = region_to_rows[r1] | region_to_rows[r2]
                if len(rows) == 2:
                    for row in rows:
                        for col in range(n):
                            reg = self.board[row][col]
                            if reg != r1 and reg != r2:
                                model.Add(x[row][col] == 0)

    def get_possible_cells_for_hinting(self):
        """Return a dict mapping region -> set of possible (row, col) cells after heuristics."""
        n = self.n
        region_positions = {}
        regions_rows = [set() for _ in range(n)]
        regions_cols = [set() for _ in range(n)]
        for row in range(n):
            for col in range(n):
                region = self.board[row][col]
                if region not in region_positions:
                    region_positions[region] = []
                region_positions[region].append((row, col))
                regions_rows[region].add(row)
                regions_cols[region].add(col)
        # Copy of region_positions for possible cells
        possible_cells = {region: set(cells) for region, cells in region_positions.items()}
        assigned_queens = set()
        changed = True
        while changed:
            changed = False
            # Singleton regions
            for region, cells in possible_cells.items():
                if len(cells) == 1:
                    (row, col) = next(iter(cells))
                    if (region, row, col) not in assigned_queens:
                        assigned_queens.add((region, row, col))
                        changed = True
                        # Eliminate all other cells in region
                        for (r, c) in list(possible_cells[region]):
                            if (r, c) != (row, col):
                                possible_cells[region].remove((r, c))
                        # Eliminate all other cells in row/col for all regions
                        for reg2 in possible_cells:
                            if reg2 == region:
                                continue
                            for (r, c) in list(possible_cells[reg2]):
                                if r == row or c == col:
                                    possible_cells[reg2].remove((r, c))
                        # Eliminate all cells in 3x3 vicinity for all regions
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                ni, nj = row + dr, col + dc
                                if 0 <= ni < n and 0 <= nj < n:
                                    for reg2 in possible_cells:
                                        if reg2 == region:
                                            continue
                                        if (ni, nj) in possible_cells[reg2]:
                                            possible_cells[reg2].remove((ni, nj))
            # Row/col singleton for region
            for region in range(n):
                if region in possible_cells:
                    rows = set(r for (r, _) in possible_cells[region])
                    if len(rows) == 1:
                        row = next(iter(rows))
                        for (r, c) in list(possible_cells[region]):
                            if r != row:
                                possible_cells[region].remove((r, c))
                    cols = set(c for (_, c) in possible_cells[region])
                    if len(cols) == 1:
                        col = next(iter(cols))
                        for (r, c) in list(possible_cells[region]):
                            if c != col:
                                possible_cells[region].remove((r, c))
        return possible_cells

    def count_unique_neighbor_regions(self, row, col):
        n = self.n
        unique_regions = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                ni, nj = row + dr, col + dc
                if 0 <= ni < n and 0 <= nj < n:
                    unique_regions.add(self.board[ni][nj])
        return len(unique_regions)

    def count_likely_queen_neighbors(self, row, col, possible_cells):
        n = self.n
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                ni, nj = row + dr, col + dc
                if 0 <= ni < n and 0 <= nj < n:
                    for cells in possible_cells.values():
                        if (ni, nj) in cells:
                            count += 1
        return count

    def region_size(self, region):
        return sum(self.board[i][j] == region for i in range(self.n) for j in range(self.n))

    def solve(self):
        """
        Solve the N-Queens problem with region constraints using CP-SAT.
        """
        model = cp_model.CpModel()
        n = self.n
        x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
        
        self.add_row_constraints(model, x)
        self.add_column_constraints(model, x)
        self.add_vicinity_constraints(model, x)
        self.add_region_constraints(model, x)
        if self.use_heuristic:
            self.add_heuristic_constraints(model, x)
            self.add_pairwise_region_column_row_heuristic(model, x)
            
            possible_cells = self.get_possible_cells_for_hinting()
            for region, cells in possible_cells.items():
                if 1 < len(cells) < 5:
                    scores = {}
                    region_sz = self.region_size(region)
                    for (row, col) in cells:
                        score = 0
                        
                        if sum(1 for (r, c) in cells if r == row) == 1:
                            score += 2
                        
                        if sum(1 for (r, c) in cells if c == col) == 1:
                            score += 2
                        
                        diversity = self.count_unique_neighbor_regions(row, col)
                        score += diversity
                        
                        threat = self.count_likely_queen_neighbors(row, col, possible_cells)
                        score -= threat
                        
                        center = (n-1)/2
                        dist = abs(row - center) + abs(col - center)
                        score -= dist * 0.5
                        
                        score -= region_sz * 0.2
                        scores[(row, col)] = score
                    sorted_cells = sorted(cells, key=lambda rc: -scores[rc])
                    max_score = scores[sorted_cells[0]]
                    for (row, col) in sorted_cells:
                        if scores[(row, col)] == max_score:
                            model.AddHint(x[row][col], 1)
                        else:
                            break
        
        flat_vars = [x[i][j] for i in range(n) for j in range(n)]
        model.AddDecisionStrategy(flat_vars,
                                 cp_model.CHOOSE_MIN_DOMAIN_SIZE,
                                 cp_model.SELECT_MIN_VALUE)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            solution = [[int(solver.Value(x[i][j])) for j in range(n)] for i in range(n)]
            return solution
        else:
            return None

    def has_one_solution(self):
        """
        Return True if the queens puzzle has exactly one solution (CP-SAT version)
        """
        model = cp_model.CpModel()
        n = self.n
        x = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(model.NewBoolVar(f'x_{i}_{j}'))
            x.append(row)

        self.add_row_constraints(model, x)
        self.add_column_constraints(model, x)
        self.add_vicinity_constraints(model, x)
        self.add_region_constraints(model, x)

        solver = cp_model.CpSolver()
        solutions = []
        class SolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self, x_vars):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.solutions = []
                self.x_vars = x_vars
            def on_solution_callback(self):
                solution = [[int(self.Value(self.x_vars[i][j])) for j in range(n)] for i in range(n)]
                self.solutions.append(solution)
                if len(self.solutions) >= 2:
                    self.StopSearch()
        collector = SolutionCollector(x)
        solver.SearchForAllSolutions(model, collector)
        self.solution = collector.solutions[0] if collector.solutions else None
        return len(collector.solutions) == 1

    def print_solution(self):
        """Pretty print the final solution"""
        if self.solution is None:
            print("No solution exists.")
            return
        
        for row in self.solution:
            print(' '.join('Q' if cell == 1 else '.' for cell in row))

if __name__ == "__main__":
    # Example usage and test
    # Sample 5x5 board with regions (0-4)
    # board = [
    #     [0, 0, 4, 4, 4],
    #     [0, 0, 4, 3, 2],
    #     [0, 2, 2, 3, 2],
    #     [1, 2, 2, 2, 2],
    #     [2, 2, 2, 2, 2]
    # ]
    # solver = CpSatSolver(board)
    # solution = solver.solve()
    # print(solution)

    test_boards = create_test_boards()
    for board in test_boards:
        solver = CpSatSolver(board)
        solution = solver.solve()
        solver.print_solution()
