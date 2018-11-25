import numpy as np
import cv2 as cv

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

point_color = np.random.randint(0,255,(100,3))

p0 = None
p_arr = None

def get_points(old_frame, new_frame):
	global p0

	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

	if (p0 is None):
		p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

	res = p1[st == 1]
	p0 = res.reshape(-1,1,2)

	return res

def get_next_points(old_frame, new_frame, start_points=[], clean=False):
	global p_arr

	if (clean):
		p_arr = None
		return []

	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

	if (p_arr == None and start_points != []):
		p_arr = start_points[:]
	else:
		return []

	p_new, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p_arr, None, **lk_params)

	res = p_new[st == 1]
	p_arr = res.reshape(-1,1,2)

	return res
