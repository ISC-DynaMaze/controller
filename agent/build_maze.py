from pathlib import Path

import cv2
from spade.behaviour import OneShotBehaviour

from agent.send_maze import SendMazeBehaviour
from walls.wall_detection import build_maze_from_path

# Behaviour to build maze from photo, save debug image, and send maze data to requester using SendMazeBehaviour
class BuildMazeBehaviour(OneShotBehaviour):
    def __init__(self, photo_path: Path, request_jid: str, output_dir: Path):
        super().__init__()
        self.photo_path = photo_path
        self.request_jid = request_jid
        self.output_dir = output_dir

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = build_maze_from_path(
                image_path=self.photo_path,
                rows=3,
                cols=11,
                kernel_len=25,
                min_length=30,
                overlap_ratio=0.6,
                cell_size=140,
                margin=40,
                wall_thickness=4,
            )
        except Exception as e:
            self.agent.logger.error(f"Failed to build maze from {self.photo_path}: {e}")
            return

        maze = result["maze"]
        # store maze in agent for later use
        self.agent.maze = maze

        # debug image
        grid_img = result["grid_img"]
        maze_img_path = self.output_dir / f"maze_{self.photo_path.stem}.jpg"
        cv2.imwrite(str(maze_img_path), grid_img)

        self.agent.logger.info(f"Maze built from {self.photo_path}")
        self.agent.logger.info(f"Debug maze image saved at {maze_img_path}")

        # send maze to requester 
        send_maze = SendMazeBehaviour(
            request_jid=self.request_jid,
            maze=maze,
        )
        self.agent.add_behaviour(send_maze)