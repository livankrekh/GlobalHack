import cv2 as cv
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

def slice_contours(contours, line):
	x_coords, y_coords = zip(*line)
	A = vstack([x_coords,ones(len(x_coords))]).T
	a, b = lstsq(A, y_coords)[0]
	
	left = []
	right = []

	for c in contours:
		M = cv.moments(c)
		if (M["m00"] != 0):
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			continue

		current_x = int((cX - b) / a)

		if ()

		if (cX < current_x):
			left.append(c)
		else:
			right.append(c)

	return left,right


def analyze_vertical(line, frame):
	compare = []

	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	_, bw = cv.threshold(gray, 80, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
	im, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

	left, right = slice_contours(contours, line)

	for left_c in left:
		for right_c in right:
			ret = cv.matchShapes(left_c, right_c, 1, 0.0)

			if (ret < 0.2):
				compare.append((left_c, right_c))
				break

	cv.imshow("Edges", im)

	return compare 
