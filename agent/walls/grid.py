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
            self.h_walls[row][col] = True
            neighbor_row, neighbor_col = row - 1, col
            opposite = DOWN

        elif direction == RIGHT:
            self.v_walls[row][col + 1] = True
            neighbor_row, neighbor_col = row, col + 1
            opposite = LEFT

        elif direction == DOWN:
            self.h_walls[row + 1][col] = True
            neighbor_row, neighbor_col = row + 1, col
            opposite = UP

        elif direction == LEFT:
            self.v_walls[row][col] = True
            neighbor_row, neighbor_col = row, col - 1
            opposite = RIGHT

        else:
            return

        if self.is_valid_cell(neighbor_row, neighbor_col):
            self.grid[neighbor_row][neighbor_col].add_wall(opposite)

    # clear all walls from the maze
    def clear_walls(self):
        self.h_walls = [[False for _ in range(self.n_cols)] for _ in range(self.n_rows + 1)]
        self.v_walls = [[False for _ in range(self.n_cols + 1)] for _ in range(self.n_rows)]

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                self.grid[row][col].walls = [False, False, False, False]

    # add outer border walls to the maze
    def add_outer_border(self):
        # top and bottom borders
        for col in range(self.n_cols):
            self.h_walls[0][col] = True
            self.h_walls[self.n_rows][col] = True

        # left and right borders
        for row in range(self.n_rows):
            self.v_walls[row][0] = True
            self.v_walls[row][self.n_cols] = True

        # update cell wall flags too
        for col in range(self.n_cols):
            self.grid[0][col].add_wall(UP)
            self.grid[self.n_rows - 1][col].add_wall(DOWN)

        for row in range(self.n_rows):
            self.grid[row][0].add_wall(LEFT)
            self.grid[row][self.n_cols - 1].add_wall(RIGHT)


    # helper functions for building maze from detected lines

    def _clamp(self, value, low, high):
        return max(low, min(high, value))

    def _snap_x_to_col_boundary(self, x, rect_x, cell_w):
        return int(round((x - rect_x) / cell_w))

    def _snap_y_to_row_boundary(self, y, rect_y, cell_h):
        return int(round((y - rect_y) / cell_h))

    def _overlap_length(self, a1, a2, b1, b2):
        left = max(a1, b1)
        right = min(a2, b2)
        return max(0, right - left)

    # build maze walls from detected pixel segments
    # overlap_ratio: minimum fraction of a cell boundary that must be covered to mark that wall as present
    def build_from_detected_lines(self, rect, horizontal_lines, vertical_lines, overlap_ratio=0.6):
        self.clear_walls()
        self.add_outer_border()

        # dimensions of the main rectangle containing the maze
        rect_x, rect_y, rect_w, rect_h = rect
        cell_w = rect_w / self.n_cols
        cell_h = rect_h / self.n_rows

        # horizontal lines -> h_walls
        for x1, y, x2, _ in horizontal_lines:
            if x2 < x1:
                x1, x2 = x2, x1

            # snap y to nearest row boundary
            row_boundary = self._snap_y_to_row_boundary(y, rect_y, cell_h)
            row_boundary = self._clamp(row_boundary, 0, self.n_rows)

            # check overlap with each cell boundary in that row
            for col in range(self.n_cols):
                cell_x1 = rect_x + col * cell_w
                cell_x2 = rect_x + (col + 1) * cell_w

                overlap = self._overlap_length(x1, x2, cell_x1, cell_x2)
                cell_width = cell_x2 - cell_x1

                # if overlap is large enough, mark the wall as present
                if overlap >= overlap_ratio * cell_width:
                    self.h_walls[row_boundary][col] = True

                    if row_boundary > 0:
                        self.grid[row_boundary - 1][col].add_wall(DOWN)
                    if row_boundary < self.n_rows:
                        self.grid[row_boundary][col].add_wall(UP)

        # vertical lines -> v_walls
        for x, y1, _, y2 in vertical_lines:
            if y2 < y1:
                y1, y2 = y2, y1

            col_boundary = self._snap_x_to_col_boundary(x, rect_x, cell_w)
            col_boundary = self._clamp(col_boundary, 0, self.n_cols)

            for row in range(self.n_rows):
                cell_y1 = rect_y + row * cell_h
                cell_y2 = rect_y + (row + 1) * cell_h

                overlap = self._overlap_length(y1, y2, cell_y1, cell_y2)
                cell_height = cell_y2 - cell_y1

                if overlap >= overlap_ratio * cell_height:
                    self.v_walls[row][col_boundary] = True

                    if col_boundary > 0:
                        self.grid[row][col_boundary - 1].add_wall(RIGHT)
                    if col_boundary < self.n_cols:
                        self.grid[row][col_boundary].add_wall(LEFT)

    # draw the maze grid with walls and cell labels for visualization
    def draw(self, cell_size=100, margin=40, wall_thickness=3):
        # calculate image size
        img_h = self.n_rows * cell_size + 2 * margin
        img_w = self.n_cols * cell_size + 2 * margin

        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        # light cell grid + labels
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x1 = margin + col * cell_size
                y1 = margin + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                cv.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 1)

                label = f"{row},{col}"
                cv.putText(
                    img,
                    label,
                    (x1 + 12, y1 + cell_size // 2),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (150, 150, 150),
                    1,
                    cv.LINE_AA
                )

        # horizontal walls
        for r in range(self.n_rows + 1):
            for c in range(self.n_cols):
                if self.h_walls[r][c]:
                    x1 = margin + c * cell_size
                    x2 = margin + (c + 1) * cell_size
                    y = margin + r * cell_size
                    cv.line(img, (x1, y), (x2, y), (0, 180, 0), wall_thickness)

        # vertical walls
        for r in range(self.n_rows):
            for c in range(self.n_cols + 1):
                if self.v_walls[r][c]:
                    x = margin + c * cell_size
                    y1 = margin + r * cell_size
                    y2 = margin + (r + 1) * cell_size
                    cv.line(img, (x, y1), (x, y2), (0, 0, 255), wall_thickness)

        return img

    # convert maze to a dictionary format for easy serialization
    def to_dict(self):
        return {
            "rows": self.n_rows,
            "cols": self.n_cols,
            "h_walls": self.h_walls,
            "v_walls": self.v_walls,
        }

    # create a maze object from a dictionary
    @classmethod
    def from_dict(cls, data):
        maze = cls(rows=data["rows"], cols=data["cols"])
        maze.clear_walls()

        maze.h_walls = data["h_walls"]
        maze.v_walls = data["v_walls"]

        for row in range(maze.n_rows):
            for col in range(maze.n_cols):
                maze.grid[row][col].walls = [False, False, False, False]

                if maze.h_walls[row][col]:
                    maze.grid[row][col].add_wall(UP)
                if maze.h_walls[row + 1][col]:
                    maze.grid[row][col].add_wall(DOWN)
                if maze.v_walls[row][col]:
                    maze.grid[row][col].add_wall(LEFT)
                if maze.v_walls[row][col + 1]:
                    maze.grid[row][col].add_wall(RIGHT)

        return maze

    def print_maze(self):
        for row in self.grid:
            for cell in row:
                print(cell)

