import logging
from pathlib import Path

from spade.agent import Agent
from spade.template import Template

from agent.photo import ReceivePhotoBehaviour, RequestPhotoBehaviour
from agent.calibration_request import CalibrationResponderBehaviour

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Enable SPADE and XMPP specific logging
for log_name in ["spade", "aioxmpp", "xmpp"]:
    log = logging.getLogger(log_name)
    log.setLevel(logging.DEBUG)
    log.propagate = True


class ControllerAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("ControllerAgent")

    async def setup(self):
        ask_photo = RequestPhotoBehaviour("camera_agent@isc-coordinator.lan")
        receive_photo = ReceivePhotoBehaviour(save_dir=Path("photos"))
        ask_angle = CalibrationResponderBehaviour()

        #Filter template
        calib_template = Template()
        calib_template.set_metadata("ontology", "calibration")

        #self.add_behaviour(ask_photo)
        #self.add_behaviour(receive_photo)
        self.add_behaviour(ask_angle, calib_template)