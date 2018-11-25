from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from stitch_api import *
from get_points import *
from symmetry import *
import sys

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    return (int(p[0]), int(p[1])), (int(q[0]), int(q[1]))

def getOrientationFromArr(data_pts, img):

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    if (cntr[0] < 0 or cntr[1] < 0):
        cntr = (0, 0)
    
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * (eigenvectors[1,0] if (len(eigenvectors) >= 2) else 1) ** 2, cntr[1] - 0.02 * (eigenvectors[1,1] if (len(eigenvectors) >= 2 and len(eigenvectors[1]) >= 2) else 1) * (eigenvalues[1,0] if (len(eigenvectors) >= 2) else 1))

    if (p1[0] < 0 or p1[1] < 0):
        p1 = (0, 0)

    if (p2[0] < 0 or p2[1] < 0):
        p2 = (0, 0)

    line_x = drawAxis(img, cntr, p1, (0, 255, 0), 1)
    line_y = drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    print(p1, p2)
    return angle, line_x, line_y, cntr
 
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * (eigenvectors[1,0] if (len(eigenvectors) >= 2) else 1) ** 2, cntr[1] - 0.02 * (eigenvectors[1,1] if (len(eigenvectors) >= 2 and len(eigenvectors[1]) >= 2) else 1) * (eigenvalues[1,0] if (len(eigenvectors) >= 2) else 1))
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle

def orientation_on_frame(src):
    if src is None:
        print('Could not open or find the image: ', sys.argv[1])
        exit(0)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        area = cv.contourArea(c);
        if area < 1e2 or 1e5 < area:
            continue
        cv.drawContours(src, contours, i, (0, 0, 255), 2);
        getOrientation(c, src)
    cv.imshow('output', src)

def arr_to_contour(arr):
    res = []

    for elem in arr:
        res.append(elem)

    return res

if __name__ == "__main__":
	
    cap = cv.VideoCapture(sys.argv[1])
    
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    status, old = cap.read()
    status, frame = cap.read()
    stitch_arr = []

    while (status):
        # Main points location calculating
        points_arr = get_points(old, frame)
        pointed = frame.copy()

        for i, elem in enumerate(points_arr):
            x,y = elem.ravel()

            if (x < 0 or y < 0):
                points_arr[i][0] = 0
                points_arr[i][1] = 0
                continue

            # cv.circle(pointed, (x,y), 5, point_color[i].tolist(), -1)

        # End calculating

        angle, line_x, line_y, mean = getOrientationFromArr(points_arr, pointed)

        if ("--standart" in sys.argv):
            analyze_standart(pointed, mean, line_x, line_y, angle)
        if ("--dynamic-symmetry" in sys.argv):
            analyze_dinamic_symmetry(pointed, line_x, line_y)
        if ("--symmetry" in sys.argv):
            analyze_symmetry(pointed, line_x, line_y, mean)

        cv.imshow('pointed', pointed)
        old = frame.copy()
        status, frame = cap.read()

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()
