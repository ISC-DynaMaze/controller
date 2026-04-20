import numpy as np
import cv2 as cv

from agent.obstacles.obstacles import yellowObstacle, redObstacle, greenObstacle

image_path = "images/maze_obj.png"

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

def main():
    image = get_image(image_path)
    #maze = None 

    y_obstacle = yellowObstacle() 
    r_obstacle = redObstacle() 
    g_obstacle = greenObstacle() 

    mask = get_obstacle_mask(image, g_obstacle.color_range)
    highlighted = highlight_obstacles(image, mask)
    cv.imshow("Obstacle Mask", mask)
    cv.imshow("Highlighted Obstacles", highlighted)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()