import numpy as np
import cv2 as cv

# wall indices
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class Cell():
    def __init__(self, row, col, walls = None):
        self.row = row
        self.col = col
        self.walls = walls if walls is not None else [False, False, False, False]

    def add_wall(self, position):
        self.walls[position] = True
    
    def has_wall(self, position):
        return self.walls[position]
    
    def __repr__(self):
        return f"Cell({self.row}, {self.col}, walls={self.walls})"


class Maze():
    def __init__(self, rows, cols):
        self.n_rows = rows
        self.n_cols = cols

        # grid of cells
        self.grid = [[Cell(i, j) for j in range(cols)] for i in range(rows)]

        # explicit wall maps
        # horizontal walls: (rows + 1) x cols
        self.h_walls = [[False for _ in range(cols)] for _ in range(rows + 1)]

        # vertical walls: rows x (cols + 1)
        self.v_walls = [[False for _ in range(cols + 1)] for _ in range(rows)]

    # check if a cell is within boundaries
    # should return true if valid cell
    def is_valid_cell(self, row, col):
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols
    
    # return cell at given row, check if it is a valid cell
    def get_cell(self, row, col):
        if self.is_valid_cell(row, col):
            return self.grid[row][col]
        else:
            # if cell not valid
            return None
        
    # add a wall to the cell and update neighboring cell's wall as well
    def add_wall(self, row, col, direction):
        if not self.is_valid_cell(row, col):
            return

        self.grid[row][col].add_wall(direction)

        if direction == UP:
            neighbor_row, neighbor_col = row - 1, col
            opposite = DOWN
        elif direction == RIGHT:
            neighbor_row, neighbor_col = row, col + 1
            opposite = LEFT
        elif direction == DOWN:
            neighbor_row, neighbor_col = row + 1, col
            opposite = UP
        elif direction == LEFT:
            neighbor_row, neighbor_col = row, col - 1
            opposite = RIGHT
        else:
            return

        if self.is_valid_cell(neighbor_row, neighbor_col):
            self.grid[neighbor_row][neighbor_col].add_wall(opposite)


    def print_maze(self):
        for row in self.grid:
            for cell in row:
                print(cell)

