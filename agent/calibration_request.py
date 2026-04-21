import json

from spade.behaviour import CyclicBehaviour
from spade.template import Template

class CalibrationResponderBehaviour(CyclicBehaviour):
    async def run(self):
        # Ce comportement ne s'activera QUE pour les messages de calibration
        msg = await self.receive(timeout=10)
        
        if msg:
            self.agent.logger.info(f"Demande de calibration reçue de {msg.sender}")
            
            # 1. On cherche l'angle en mémoire (ici on suppose que ton robot a l'ID ArUco n°1)
            # Si tu as plusieurs robots, il faudra mapper msg.sender avec l'ID ArUco !
            id_aruco_du_robot = 7
            
            angle_actuel = 0.0
            if hasattr(self.agent, "known_angles") and id_aruco_du_robot in self.agent.known_angles:
                angle_actuel = self.agent.known_angles[id_aruco_du_robot]
            else:
                self.agent.logger.warning("[Warning] Robot not on the camera")
            
            # 2. On fabrique la réponse pour le robot
            reply = msg.make_reply()
            reply.set_metadata("performative", "inform")
            reply.set_metadata("ontology", "calibration")
            
            # On envoie EXACTEMENT ce que le robot attend : {"angle": 45.0}
            data = {"action": "bot-rot", "id": id_aruco_du_robot, "angle": angle_actuel}
            reply.body = json.dumps(data)
            
            await self.send(reply)
            self.agent.logger.debug(f"Réponse envoyée au robot : {reply.body}")