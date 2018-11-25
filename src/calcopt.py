import numpy as np
import cv2

from stitch_api import stitch_api

cap = cv2.VideoCapture('hack.MOV')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

def stitching(frame_buf,points_buf):
	#show frames in buffer
	
	for i in range(0,10):
		#frame = frame_buf[i]
		#good_new = points_buf[i]
		#frame = frame_buf[i]
		#for i,(new) in enumerate(good_new):
		#a,b = new.ravel()
		#frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
		#good_new.reshape(-1,1,2)
		#cv2.imshow('frame',frame)
		#k = cv2.waitKey(1000) & 0xff
		stitch(frame_buf[0],frame_buf[i+1])
	


# Take first frame and find corners in it
ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

num=0
j=0
frame_buf = []
points_buf = []
while(1):
    num=num+1
    #for i in range(0,20):
    ret,frame = cap.read()
    buf_frame = frame.copy()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    #img = cv2.add(frame,mask)

    # make buffer of frames not each
    if num % 20 == 1: 
    	frame_buf.append(buf_frame)
    points_buf.append(good_new)
    print("ok")

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        stitch_api(frame_buf)
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

