import asyncio
import base64
from pathlib import Path

import aiofiles

from photo import ReceivePhotoBehaviour


class FakeLogger:
    def info(self, msg):
        print("[INFO]", msg)

    def error(self, msg):
        print("[ERROR]", msg)


class FakeMessage:
    def __init__(self, body):
        self.body = body


class FakeAgent:
    def __init__(self):
        self.logger = FakeLogger()
        self.behaviours = []

    def add_behaviour(self, behaviour):
        self.behaviours.append(behaviour)
        print(f"[INFO] Added behaviour: {type(behaviour).__name__}")


class TestReceivePhotoBehaviour(ReceivePhotoBehaviour):
    def __init__(self, save_dir: Path, maze_dir: Path, request_jid: str, fake_msg):
        super().__init__(save_dir=save_dir, maze_dir=maze_dir, request_jid=request_jid)
        self.fake_msg = fake_msg

    async def receive(self, timeout=None):
        return self.fake_msg


async def main():
    image_path = Path("images/maze4.png")

    async with aiofiles.open(image_path, "rb") as f:
        img_data = await f.read()

    encoded_img = base64.b64encode(img_data).decode("utf-8")
    fake_msg = FakeMessage(body=encoded_img)

    fake_agent = FakeAgent()

    behaviour = TestReceivePhotoBehaviour(
        save_dir=Path("photos"),
        maze_dir=Path("mazes"),
        request_jid="receiver@fake.local",
        fake_msg=fake_msg,
    )
    behaviour.agent = fake_agent

    await behaviour.run()

    print("\nBehaviours added to fake agent:")
    for b in fake_agent.behaviours:
        print("-", type(b).__name__)


if __name__ == "__main__":
    asyncio.run(main())