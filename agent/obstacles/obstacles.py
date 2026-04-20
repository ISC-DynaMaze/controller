from agent.walls.grid import Cell

class Obstacle:
    def __init__(self):
        self.cell = None
        #self.color = (0,0,0)
        self.color_range = [[(0,0,0), (0,0,0)]]


    
class greenObstacle(Obstacle): 
    def __init__(self):
        super().__init__()
        self.color_range = [[(35, 50, 50), (85, 255, 255)]]

class redObstacle(Obstacle):
    def __init__(self):
        super().__init__()
        # red uses two ranges in HSV
        self.color_range = [[(0, 80, 80), (5, 255, 255)], [(178, 80, 80), (185, 255, 255)]]

class yellowObstacle(Obstacle):
    def __init__(self):
        super().__init__()
        self.color_range = [[(20, 80, 80), (35, 255, 255)]]