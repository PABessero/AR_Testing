import numpy as np
from mss import mss
from PIL import Image
import argparse
import imutils
import time
import cv2
import sys
from config import ARUCO_DICT

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
                default="DICT_ARUCO_ORIGINAL",
                help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

print("[INFO] Detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

time.sleep(0.2)

mon = {'left': 1920, 'top': 0, 'width': 1920, 'height': 1080}

sct = mss()

while True:
    frame = sct.grab(mon)
    frame = Image.frombytes('RGB', (frame.width, frame.height), frame.rgb)
    frame = np.array(frame)

    (corners, ids, rejectedImgPoints) = arucoDetector.detectMarkers(frame)

    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[INFO] ArUCo marker ID: {}".format(markerID))

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
