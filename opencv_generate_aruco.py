import numpy as np
import argparse
import cv2
import sys
from config import ARUCO_DICT

ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output", required=True,
                help="path to output image containing ArUCo tag")
ap.add_argument("-i", "--id", required=True, type=int,
                help="ID of ArUCo tag to generate")
ap.add_argument("-t", "--type", type=str,
                default="DICT_ARUCO_ORIGINAL",
                help="Type of ArUCo tag to generate")

args = vars(ap.parse_args())

if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not defined".format(args["type"]))
    sys.exit()

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])

print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(args["type"], args["id"]))

tag = np.zeros((300, 300, 1), dtype=np.uint8)
cv2.aruco.generateImageMarker(arucoDict, args["id"], 300, tag, 1)

cv2.imwrite(args["output"], tag)
cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)