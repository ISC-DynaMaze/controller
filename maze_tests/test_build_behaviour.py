import asyncio
from pathlib import Path

from build_maze import BuildMazeBehaviour

class FakeLogger:
    def info(self, msg):
        print("[INFO]", msg)

    def error(self, msg):
        print("[ERROR]", msg)


class FakeAgent:
    def __init__(self):
        self.logger = FakeLogger()
        self.behaviours = []

    def add_behaviour(self, behaviour):
        self.behaviours.append(behaviour)
        print(f"[INFO] Added behaviour: {type(behaviour).__name__}")


async def main():
    fake_agent = FakeAgent()

    behaviour = BuildMazeBehaviour(
        photo_path=Path("images/maze4.png"),
        request_jid="receiver@fake.local",
        output_dir=Path("mazes"),
    )
    behaviour.agent = fake_agent

    await behaviour.run()

    print("\nBehaviours added to fake agent:")
    for b in fake_agent.behaviours:
        print("-", type(b).__name__)


if __name__ == "__main__":
    asyncio.run(main())