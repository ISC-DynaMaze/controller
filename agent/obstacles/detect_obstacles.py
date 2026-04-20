import numpy as np
import cv2 as cv

from agent.obstacles.obstacles import yellowObstacle, redObstacle, greenObstacle
from agent.walls.wall_detection import build_maze_from_path

# get maze from agent
def get_built_maze(agent):
    return getattr(agent, "maze", None)

def get_image(image_path):
    img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Error opening image: {image_path}")
    return img

# get a mask for given color
def get_obstacle_mask(image, color_range):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    print(f"HSV: {hsv[154:170, 60:80]}")

    # one color range (yellow and green) 
    if len(color_range) == 1:
        lower, upper = color_range[0]
        mask = cv.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    # red that uses 2 color ranges
    else:
        lower_1, upper_1 = color_range[0]
        lower_2, upper_2 = color_range[1]
        mask_1 = cv.inRange(hsv, np.array(lower_1, dtype=np.uint8), np.array(upper_1, dtype=np.uint8))
        mask_2 = cv.inRange(hsv, np.array(lower_2, dtype=np.uint8), np.array(upper_2, dtype=np.uint8))
        mask = cv.bitwise_or(mask_1, mask_2)

    # small cleanup to close gaps in obstacles
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    return mask 

# higjlight detected objects
def highlight_obstacles(image, mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    highlighted = image.copy()
    cv.drawContours(highlighted, contours, -1, (0, 255, 255), 2)  # yellow contours
    return highlighted

# get the center of the detected obstacles, will be used to determine their position in the grid 
def get_obstacle_center(mask, min_area=100):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centers = []

    for contour in contours: 
        area = cv.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append({"contour": contour, "bbox": (x, y, w, h), "center": (center_x, center_y), "area": area})

    return centers

def main():
    image_path = "images/maze_obj.png"
    image = get_image(image_path)
    #maze = get_built_maze(None)  #TODO replace with agent

    # get maze from image - normally would have to take maze from agent 
    result = build_maze_from_path(
                image_path=image_path,
                rows=3,
                cols=11,
                kernel_len=25,
                min_length=30,
                overlap_ratio=0.6,
                cell_size=140,
                margin=40,
                wall_thickness=4,
            )
 
    maze = result["maze"]

    y_obstacle = yellowObstacle() 
    r_obstacle = redObstacle() 
    g_obstacle = greenObstacle() 

    mask = get_obstacle_mask(image, g_obstacle.color_range)
    highlighted = highlight_obstacles(image, mask)
    cv.imshow("Obstacle Mask", mask)
    cv.imshow("Highlighted Obstacles", highlighted)
    cv.imshow("Maze Grid", result["grid_img"])
    cv.imshow("Detected Segments", result["debug_img"])
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()