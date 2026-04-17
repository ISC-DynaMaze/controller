import json

from spade.behaviour import Message, OneShotBehaviour

# Behaviour to send maze data to requester
class SendMazeBehaviour(OneShotBehaviour):
    def __init__(self, request_jid: str, maze):
        super().__init__()
        self.request_jid = request_jid
        self.maze = maze

    async def run(self) -> None:
        data = {
            "type": "maze-data",
            "maze": self.maze.to_dict(),
        }

        msg = Message(to=self.request_jid)
        msg.set_metadata("performative", "inform")
        msg.body = json.dumps(data)

        await self.send(msg)
        self.agent.logger.info(f"Maze data sent to {self.request_jid}")