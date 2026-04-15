import cv2
from imutils.video import VideoStream
import numpy as np

def detect_markers(frame):
    corners, ids, rejected = detector.detectMarkers(frame)
    return corners, ids, rejected

def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    stream = VideoStream(0).start()
    
    obj_points = np.array([
        [-50, 50],
        [50, 50],
        [50, -50],
        [-50, -50],
    ])
    cam_matrix = np.array([
        [951.2640682, 0. , 644.98320654],
        [0., 944.24396462, 360.0181631],
        [0., 0., 1.]
    ])
    dist_coeffs = np.array([0.09431487, -0.26994709,  0.00180207,  0.00271549,  0.40307225])
    
    while True:
        frame = stream.read()
        if frame is None:
            break
        #cv2.imshow("Camera", frame)
        
        corners, ids, rejected = detector.detectMarkers(frame)
        with_markers = frame.copy()
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(with_markers, corners, ids)
            #for corner, id in zip(corners, ids):
            #    print(obj_points.shape, corner.shape)
            #    ret, rvec, tvec = cv2.solvePnP(obj_points, corner[0], cam_matrix, dist_coeffs)
            #    cv2.drawFrameAxes(with_markers, cam_matrix, dist_coeffs, rvec, tvec, 100, 2)
        cv2.imshow("Markers", with_markers)
        key = cv2.waitKey(1)
        if key != -1:
            break
    
    cv2.destroyAllWindows()
    stream.stop()


if __name__ == "__main__":
    main()
