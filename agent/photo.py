import base64
import datetime
from pathlib import Path
import aiofiles
import cv2
import numpy as np
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
class ReceivePhotoBehaviour(CyclicBehaviour):
    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    async def run(self):
        msg = await self.receive(timeout=10)
        if msg and msg.body:
            # Logique de sauvegarde uniquement
            img_data = base64.b64decode(msg.body)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"archive_{timestamp}.jpg"

            async with aiofiles.open(filepath, "wb") as img_file:
                await img_file.write(img_data)
            
            print(f"Photo archivée : {filepath}")