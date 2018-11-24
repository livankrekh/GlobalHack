from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import sys
 
def stitch_api(imgs):

	 
	stitcher = cv.createStitcher(False)
	status, pano = stitcher.stitch(imgs)
	 
	if status != cv.Stitcher_OK:
	    print("Can't stitch images, error code = %d" % status)
	    sys.exit(-1)
	
	# cv.imshow('frame',pano)
	k = cv.waitKey(0) & 0xff
	#cv.imwrite(args.output, pano);
	print("stitching completed successfully. %s saved!" )
	return pano





