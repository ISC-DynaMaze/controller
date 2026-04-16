import numpy as np
import cv2 as cv

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

    return mask

# function to detect lines, min line length should not be too big bc some walls can be short
def detect_lines(mask):
    lines = cv.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=20
    )

    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((x1, y1, x2, y2))

    return detected_lines

# split vertical and horizontal lines 
def split_horizontal_vertical(lines, angle_tolerance=10):
    horizontal = []
    vertical = []

    # check angle of each line, if close to 0 or 180 -> horizontal, if close to 90 -> vertical
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(np.arctan2(dy, dx))
        angle = abs(angle)

        # horizontal
        if angle < angle_tolerance or angle > 180 - angle_tolerance:
            horizontal.append((x1, y1, x2, y2))

        # vertical
        elif abs(angle - 90) < angle_tolerance:
            vertical.append((x1, y1, x2, y2))

    return horizontal, vertical

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

# merge lines into one line if they are close and collinear -> will help to create the maze's walls
def merge_close_lines(lines, orientation, pos_threshold=15, gap_threshold=20):
    if not lines:
        return []

    merged = []

    if orientation == "horizontal":
        # sort by y, then x
        lines = sorted(lines, key=lambda line: (line[1], line[0]))

        for x1, y1, x2, y2 in lines:
            if x1 > x2:
                x1, x2 = x2, x1

            added = False

            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                # same horizontal level and touching/overlapping
                if abs(y1 - my1) <= pos_threshold and x1 <= mx2 + gap_threshold:
                    new_x1 = min(mx1, x1)
                    new_x2 = max(mx2, x2)
                    new_y = int(round((my1 + y1) / 2))
                    merged[i] = (new_x1, new_y, new_x2, new_y)
                    added = True
                    break

            if not added:
                merged.append((x1, y1, x2, y2))

    elif orientation == "vertical":
        # sort by x, then y
        lines = sorted(lines, key=lambda line: (line[0], line[1]))

        for x1, y1, x2, y2 in lines:
            if y1 > y2:
                y1, y2 = y2, y1

            added = False

            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                # same vertical level and touching/overlapping
                if abs(x1 - mx1) <= pos_threshold and y1 <= my2 + gap_threshold:
                    new_y1 = min(my1, y1)
                    new_y2 = max(my2, y2)
                    new_x = int(round((mx1 + x1) / 2))
                    merged[i] = (new_x, new_y1, new_x, new_y2)
                    added = True
                    break

            if not added:
                merged.append((x1, y1, x1, y2))

    else:
        # error if no orientation
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    return merged


def main():
    img = get_image(image_path)
    mask = get_pink_mask(image_path)

    lines = detect_lines(mask)
    horizontal, vertical = split_horizontal_vertical(lines)

    rect = find_outer_rectangle(mask)

    horizontal_in = keep_lines_in_rectangle(horizontal, rect)
    vertical_in = keep_lines_in_rectangle(vertical, rect)

    horizontal_merged = merge_close_lines(horizontal_in, "horizontal")
    vertical_merged = merge_close_lines(vertical_in, "vertical")

    result = img.copy()
    result = draw_outer_rectangle(result, rect)
    result = draw_lines(result, horizontal_merged, vertical_merged)

    # see how many final lines
    print(f"Horizontal lines: {len(horizontal_in)} -> merged: {len(horizontal_merged)}")
    print(f"Vertical lines: {len(vertical_in)} -> merged: {len(vertical_merged)}")

    cv.imshow("Mask", mask)
    cv.imshow("Merged lines", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()