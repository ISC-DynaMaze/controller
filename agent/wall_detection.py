import numpy as np
import cv2 as cv

image_path = "images/wall.jpg"


def get_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Error opening image: {image_path}")
    return img

# get a pink mask from the image using HSV color space -> binary image with white walls and black background
# TODO: test on actual maze walls
def get_pink_mask(image_path):
    img = get_image(image_path)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow("HSV Image", hsv)

    # lower and upper bounds for pink 
    lower = np.array([140, 40, 40], dtype=np.uint8)
    upper = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower, upper)
    return mask


def main():
    img = get_image(image_path)
    mask = get_pink_mask(image_path)

    # detect lines with hough transform -> returns a list of lines in the format (x1, y1, x2, y2)
    lines = cv.HoughLinesP(mask, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=20)

    result = img.copy()

    # draw lines on source image  
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv.imshow("Mask", mask)
    cv.imshow("Result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()