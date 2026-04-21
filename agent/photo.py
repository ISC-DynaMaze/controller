import base64
import datetime
from pathlib import Path

import aiofiles
import cv2
import numpy as np
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, Message, OneShotBehaviour

from agent.bot_detection import BotDetectionBehaviour


class RequestPhotoBehaviour(OneShotBehaviour):
    agent: Agent

    def __init__(self, camera_jid: str):
        super().__init__()
        self.camera_jid: str = camera_jid

    async def run(self):
        msg = Message(to=self.camera_jid)
        msg.set_metadata("performative", "request")
        msg.body = "Requesting photo"

        await self.send(msg)
        print("Request for photo sent.")


class ReceivePhotoBehaviour(CyclicBehaviour):
    agent: Agent

    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir: Path = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    async def run(self):
        print("Waiting for photo message...")
        msg = await self.receive(timeout=9999)
        if msg is not None and msg.body is not None:
            print("Received photo message.")
            img_data = base64.b64decode(msg.body)

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            filepath = self.save_dir / filename

            # Save the received image
            async with aiofiles.open(filepath, "wb") as img_file:
                await img_file.write(img_data)

            print(f"Photo saved as '{filepath}'.")
            img: np.ndarray = cv2.imread(filepath)  # type: ignore
            bot_detection = BotDetectionBehaviour(img, msg.sender_jid)
            self.agent.add_behaviour(bot_detection)
