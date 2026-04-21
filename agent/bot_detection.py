import json
import logging
from typing import Sequence

import cv2
import numpy as np
from spade.agent import Agent
from spade.behaviour import Message, OneShotBehaviour


class BotDetection:
    def __init__(self):
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

    def get_angles_from_markers(
        self, corners: Sequence[np.ndarray], ids: np.ndarray
    ) -> list[tuple[int, float]]:
        angles: list[tuple[int, float]] = []
        for corner, id in zip(corners, ids):
            tl, tr, br, bl = corner[0]
            v: np.ndarray = bl - tl
            angle = np.atan2(v[1], v[0])
            angles.append((int(id), float(np.degrees(angle))))
        return angles
