import cv2 as cv
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

def is_vertical(line, frame):
	h, w, _ = frame.shape

	return line[0][0] < w and line[1][0] > 0

def get_clean_contours(frame):
	new_contours = []

	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	_, bw = cv.threshold(gray, 50, 150, cv.THRESH_BINARY | cv.THRESH_OTSU)
	bw = cv.GaussianBlur(bw, (3,3), 0);
	# edges = cv.Canny(bw, 100, 200)
	im, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
	for i, c in enumerate(contours):
		area = cv.contourArea(c);
		if area < 1e2 or 1e5 < area:
			continue

		new_contours.append(c)

	cv.imshow("Gray", im)

	return new_contours

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
		x,y,w,h = cv.boundingRect(c)

		if (w*h < 2000):
			continue

		if (cX < current_x):
			left.append(c)
		else:
			right.append(c)

	return left,right

def analyze_symmetry(frame, line_x, line_y, mean):
	h, w, _ = frame.shape
	overlay = frame.copy()
	alpha = 0.5

	if (abs(line_x[0][0] - line_x[1][0]) <= 5 or abs(line_y[0][0] - line_y[1][0]) <= 5):
		cv.line(overlay, (mean[0], 0), (mean[0], h), (255, 255, 255), 15)
		cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
		cv.putText(frame, "Vertical symmetry was found!", (10,30), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 127, 0))
		return True

	return False

def show_standart_grid(frame):
	h, w, _ = frame.shape
	overlay = frame.copy()
	alpha = 0.5

	cv.line(overlay, (int(w / 3), 0), (int(w / 3), h), (255, 255, 255), 15)
	cv.line(overlay, (int(w / 3 * 2), 0), (int(w / 3 * 2), h), (255, 255, 255), 15)
	cv.line(overlay, (0, int(h / 3)), (w, int(h / 3)), (255, 255, 255), 15)
	cv.line(overlay, (0, int(h / 3 * 2)), (w, int(h / 3 * 2)), (255, 255, 255), 15)

	cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def get_correct_line(line_x, line_y):
	d_x = np.sqrt( (line_x[1][0] - line_x[0][0]) ** 2 + (line_x[1][1] - line_x[0][1]) ** 2 )
	d_y = np.sqrt( (line_y[1][0] - line_y[0][0]) ** 2 + (line_y[1][1] - line_y[0][1]) ** 2 )

	if (d_x >= d_y):
		return line_x
	else:
		return line_y

def analyze_standart(frame, mean, line_x, line_y, angle):
	h, w, _ = frame.shape
	line = get_correct_line(line_x, line_y)

	if (is_vertical(line, frame)):
		ax_line = [(mean[0] + int(mean[1] * np.sin(angle)), 0), (int(mean[0] - int(mean[1] * np.sin(angle)) ), h)]
	else:
		ax_line = [(0, mean[1] + int(mean[0] * np.sin(np.pi - angle))), (w, int(mean[1] - int(mean[0] * np.sin(np.pi - angle)) ))]

	if ( abs(int(w / 3 * 2) - mean[0]) <= w / 8 ):
		show_standart_grid(frame)
		if ( abs(ax_line[0][0] - int(w / 3 * 2)) > 50 and abs(ax_line[1][0] - int(w / 3 * 2)) > 50):
			overlay = frame.copy()
			alpha = 0.5

			cv.line(overlay, ax_line[0], ax_line[1], (0, 255, 0), 15)
			cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

		return True

	return False

def analyze_dinamic_symmetry(frame, line_x, line_y):
	line_corr = get_correct_line(line_x, line_y)
	contours = get_clean_contours(frame)
	new_img = frame.copy()
	lines = []

	for i, c in enumerate(contours):
		add_new = True

		rows,cols = frame.shape[:2]
		[vx, vy, x, y] = cv.fitLine(c, cv.DIST_L2, 0, 0.01, 0.01)
		lefty = int((-x * vy / vx) + y)
		righty = int(((cols - x) * vy / vx) + y)

		for i,line in enumerate(lines):
			if (abs(line[0][1] - righty) < 70 and abs(line[1][1] - lefty) < 70):
				lines[i][2] += 1
				add_new = False
				break

		if (add_new):
			lines.append([(cols - 1, righty), (0, lefty), 1])
		# cv.line(new_img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
		# cv.drawContours(new_img, contours, i, (0, 0, 255), 2)

	new_lines = []

	for line in lines:
		if line[2] >= 3:
			new_lines.append(line)

	same = []

	for i, line in enumerate(new_lines):
		break_loop = False
		an_sin = abs((line[0][0] - line[1][0]) / np.sqrt( (line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2 ) )

		for j, line_cp in enumerate(new_lines):
			if (i == j):
				continue

			an_sin2 = abs((line_cp[0][0] - line_cp[1][0]) / np.sqrt( (line_cp[0][0] - line_cp[1][0]) ** 2 + (line_cp[0][1] - line_cp[1][1]) ** 2 ) )

			if ( abs(an_sin - an_sin2) <= 0.04 ):
				if (same == []):
					same.append(line)
				
				same.append(line_cp)
				break_loop = True

		if (break_loop):
			break

	if (same != []):
		h, w, _ = frame.shape
		overlay = frame.copy()
		alpha = 0.5

		for line in same:
			cv.line(overlay, line[0], line[1], (255,255,255), 15)

		cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
		cv.putText(frame, "Dynamic symmetry was found!", (10,30), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 127, 0))
		return True

	return False

