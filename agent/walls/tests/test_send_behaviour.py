import asyncio

from send_maze import SendMazeBehaviour
from walls.grid import Maze, RIGHT


class FakeLogger:
    def info(self, msg):
        print("[INFO]", msg)

    def error(self, msg):
        print("[ERROR]", msg)


class FakeAgent:
    def __init__(self):
        self.logger = FakeLogger()


class TestSendMazeBehaviour(SendMazeBehaviour):
    async def send(self, msg):
        print("[INFO] Message sent")
        print("to:", msg.to)
        print("body:", msg.body[:500])


async def main():
    maze = Maze(3, 11)
    maze.add_outer_border()
    maze.add_wall(0, 0, RIGHT)

    behaviour = TestSendMazeBehaviour(
        request_jid="receiver@fake.local",
        maze=maze,
    )
    behaviour.agent = FakeAgent()

    await behaviour.run()


if __name__ == "__main__":
    asyncio.run(main())