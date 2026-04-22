from agent.walls.grid import Cell

class Obstacle:
    def __init__(self):
        self.cells = []  # list of cells occupied by the obstacle
        self.center = None
        self.color_range = [[(0,0,0), (0,0,0)]]

    def add_cell(self, cell):
        self.cells.append(cell)

    
class greenObstacle(Obstacle): 
    def __init__(self):
        super().__init__()
        self.color_range = [[(35, 50, 50), (85, 255, 255)]]

class redObstacle(Obstacle):
    def __init__(self):
        super().__init__()
        # red uses two ranges in HSV
        self.color_range = [[(0, 80, 80), (2, 255, 255)], [(178, 80, 80), (185, 255, 255)]]

class yellowObstacle(Obstacle):
    def __init__(self):
        super().__init__()
        self.color_range = [[(20, 80, 80), (35, 255, 255)]]