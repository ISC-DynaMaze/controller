from pathlib import Path
import cv2

from walls.wall_detection import build_maze_from_path

photo_path = Path("images/maze4.png")
output_path = Path("mazes/test_maze.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

result = build_maze_from_path(
    image_path=photo_path,
    rows=3,
    cols=11,
    kernel_len=25,
    min_length=30,
    overlap_ratio=0.6,
    cell_size=140,
    margin=40,
    wall_thickness=4,
)

maze_img = result["grid_img"]
cv2.imwrite(str(output_path), maze_img)

cv2.imshow("Maze Grid", maze_img)
cv2.imshow("Detected Segments", result["debug_img"])
cv2.imshow("Mask", result["mask"])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Maze saved to {output_path}")