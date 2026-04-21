import json
import logging
from typing import Sequence
import datetime

import cv2
import numpy as np
from spade.agent import Agent
from spade.behaviour import Message, OneShotBehaviour


class BotDetector:
    def __init__(self, save_dir : str = "debug_photos"):
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

    def get_angles(self, img, target_id: int):
        corners, ids, _ = self.detector.detectMarkers(img)

        debug_img = img.copy()
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = str(self.save_path / f"debug_{timestamp}.jpg")

        cv2.imwrite(filename, debug_img)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if int(marker_id) == target_id:
                    tl, tr, br, bl = corners[i][0]
                    v = bl - tl
                    angle_rad = np.atan2(v[1], v[0])
                    return float(np.degrees(angle_rad))

        return None
