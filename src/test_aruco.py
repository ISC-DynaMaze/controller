import cv2
import numpy as np

def detect_markers(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(frame)
    return corners, ids, rejected

def main():
    cam = cv2.VideoCapture(0)
    obj_points = np.array([
        [-50, 50, 0],
        [50, 50, 0],
        [50, -50, 0],
        [-50, -50, 0],
    ], dtype=np.float32)
    cam_matrix = np.array([
        [951.2640682, 0. , 644.98320654],
        [0., 944.24396462, 360.0181631],
        [0., 0., 1.]
    ])
    dist_coeffs = np.array([0.09431487, -0.26994709,  0.00180207,  0.00271549,  0.40307225])
    
    while True:
        ret, frame = cam.read()
        #cv2.imshow("Camera", frame)
        
        corners, ids, rejected = detect_markers(frame)
        with_markers = frame.copy()
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(with_markers, corners, ids)
            for corner, id in zip(corners, ids):
                print(obj_points.shape, corner.shape)
                ret, rvec, tvec = cv2.solvePnP(obj_points, corner, cam_matrix, dist_coeffs)
                cv2.drawFrameAxes(with_markers, cam_matrix, dist_coeffs, rvec, tvec, 100, 2)
        cv2.imshow("Markers", with_markers)
        print(corners, ids, rejected)
        key = cv2.waitKey(100)
        if key != -1:
            break


if __name__ == "__main__":
    main()
