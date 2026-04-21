import json
import base64
import cv2
import numpy as np
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from agent.bot_detection import BotDetector

class CalibrationResponderBehaviour(CyclicBehaviour):
    async def on_start(self):
        self.detector = BotDetector()
        self.current_robot_request = None

    async def run(self):
        msg = await self.receive(timeout=60)
        if not msg: return

        # ÉTAPE 1 : Le robot demande
        if msg.metadata.get("ontology") == "calibration":
            self.current_robot_request = msg
            # Demande à la caméra
            msg_cam = Message(to="camera_agent@isc-coordinator.lan")
            msg_cam.set_metadata("performative", "request")
            await self.send(msg_cam)

        # ÉTAPE 2 : La caméra répond
        elif msg.body and not msg.metadata.get("ontology"):
            if self.current_robot_request:
                try:
                    # Décodage
                    img_data = base64.b64decode(msg.body)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Utilisation de l'outil existant
                    angle = self.detector.get_angles(img, target_id=7)
                    
                    # Si pas trouvé, on peut renvoyer 0.0 ou une erreur
                    valeur_angle = angle if angle is not None else 0.0

                    # Réponse au robot
                    reply = self.current_robot_request.make_reply()
                    reply.set_metadata("ontology", "calibration")
                    reply.body = json.dumps({"angle": valeur_angle})
                    await self.send(reply)
                    
                    self.current_robot_request = None # Reset
                except Exception as e:
                    self.agent.logger.error(f"Error : {e}")