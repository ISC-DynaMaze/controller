import numpy as np
import cv2 as cv
from grid import Maze

image_path = "images/maze.jpg"


def get_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Error opening image: {image_path}")
    return img

# get a pink mask from the image using HSV color space -> binary image with white walls and black background
def get_pink_mask(image_path):
    img = get_image(image_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # lower and upper bounds for pink 
    lower = np.array([140, 40, 40], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower, upper)

    # small cleanup to close gaps in walls
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    return mask

# get main rectangle of maze
def find_outer_rectangle(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contour found.")

    biggest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(biggest_contour)

    return (x, y, w, h)

# keep only lines that are in the main rectangle
def keep_lines_in_rectangle(lines, rect, margin=10):
    x, y, w, h = rect

    x_min = x - margin
    y_min = y - margin
    x_max = x + w + margin
    y_max = y + h + margin

    kept_lines = []

    for x1, y1, x2, y2 in lines:
        if (
            x_min <= x1 <= x_max and
            x_min <= x2 <= x_max and
            y_min <= y1 <= y_max and
            y_min <= y2 <= y_max
        ):
            kept_lines.append((x1, y1, x2, y2))

    return kept_lines

# mask with horizontal lines
def extract_horizontal_mask(mask, kernel_len=25):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_len, 1))
    horizontal = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return horizontal

# mask with vertical lines
def extract_vertical_mask(mask, kernel_len=25):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_len))
    vertical = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return vertical

# get horizontal segments as (x1, y, x2, y) from horizontal mask, filter by min_length
def get_horizontal_segments(horizontal_mask, min_length=30):
    contours, _ = cv.findContours(horizontal_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    segments = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if w >= min_length:
            y_center = y + h // 2
            segments.append((x, y_center, x + w, y_center))

    segments.sort(key=lambda line: (line[1], line[0]))
    return segments

# get vertical segments as (x, y1, x, y2) from vertical mask, filter by min_length
def get_vertical_segments(vertical_mask, min_length=30):
    contours, _ = cv.findContours(vertical_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    segments = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if h >= min_length:
            x_center = x + w // 2
            segments.append((x_center, y, x_center, y + h))

    segments.sort(key=lambda line: (line[0], line[1]))
    return segments

# draw main rectangle on img
def draw_outer_rectangle(img, rect):
    x, y, w, h = rect
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return img

# draw lines on img
def draw_lines(img, horizontal, vertical):
    # horizontal in green
    for x1, y1, x2, y2 in horizontal:
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # vertical in red
    for x1, y1, x2, y2 in vertical:
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return img

# main function to detect maze walls and return segments and masks
def detect_maze_walls(image_path, kernel_len=25, min_length=30):
    # source image and mask
    img = get_image(image_path)
    mask = get_pink_mask(image_path)

    # contours of maze
    rect = find_outer_rectangle(mask)

    # masks for horizontal and vertical lines
    horizontal_mask = extract_horizontal_mask(mask, kernel_len=kernel_len)
    vertical_mask = extract_vertical_mask(mask, kernel_len=kernel_len)

    # segments of horizontal and vertical lines
    horizontal_lines = get_horizontal_segments(horizontal_mask, min_length=min_length)
    vertical_lines = get_vertical_segments(vertical_mask, min_length=min_length)

    # filter lines to keep only those in the main rectangle
    horizontal_lines = keep_lines_in_rectangle(horizontal_lines, rect)
    vertical_lines = keep_lines_in_rectangle(vertical_lines, rect)

    return rect, horizontal_lines, vertical_lines, img, mask, horizontal_mask, vertical_mask


def main():
    rect, horizontal_lines, vertical_lines, img, mask, horizontal_mask, vertical_mask = detect_maze_walls(
        image_path=image_path,
        kernel_len=25,
        min_length=30
    )

    result = img.copy()
    result = draw_outer_rectangle(result, rect)
    result = draw_lines(result, horizontal_lines, vertical_lines)

    x, y, w, h = rect
    print(f"Outer rectangle: x={x}, y={y}, w={w}, h={h}")
    print(f"Horizontal segments: {len(horizontal_lines)}")
    print(f"Vertical segments: {len(vertical_lines)}")

    maze = Maze(rows=3, cols=11)

    cv.imshow("Mask", mask)
    cv.imshow("Horizontal mask", horizontal_mask)
    cv.imshow("Vertical mask", vertical_mask)
    cv.imshow("Detected segments", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()