# imports
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
#not matplotlib for now

# reference image of product
target = cv2.imread("Products/Label.jpg")
#print(target)
#target = cv2.resize(target, None, fx=0.5, fy=0.5)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

## Detect Keypoints with SURF detector
minHessian = 400
orb = cv2.ORB_create()

keyP, Desc = orb.detectAndCompute(target_gray, None)

## Start camera rolling and compute this on the live stream
# initialise camera
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = camera.resolution)
# allow time for camera to wake up
time.sleep(1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
	# grab raw image
	scene = frame.array
	scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

	# Editing frame to detect keypoints
	keyP2, Desc2 = orb.detectAndCompute(scene_gray, None)
	"""
	LEGACY BECAUSE OF MYSTERY ERROR
	#FLANN matching
	index_params = dict(algorithm = 0, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(np.asarray(Desc,np.float32), np.asarray(Desc2,np.float32), k=2)
	"""
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(Desc, Desc2)
	matches = sorted(matches, key = lambda x:x.distance)

	#matchesMask = [[0,0] for i in range(len(matches))]

	clear_matches = matches
	"""
	for i, (m,n) in enumerate(matches):
		if m.distance < 0.6*n.distance:
			matchesMask[i] = [1,0]
			clear_matches.append(m)
	"""

	if len(clear_matches) > 10:
		src = np.float32([keyP[m.queryIdx].pt for m in clear_matches]).reshape(-1,1,2)
		dst = np.float32([keyP2[m.trainIdx].pt for m in clear_matches]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src,dst,cv2.RANSAC,5.0)

		h,w = target_gray.shape
		pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
		#print(M)
		if M is not None:
			dsts = cv2.perspectiveTransform(pts,M)
			scene_box = cv2.polylines(scene_gray, [np.int32(dsts)], True, 255, 3, cv2.LINE_AA)
		else:
			scene_box = scene_gray
	else:
		scene_box = scene_gray

	draw_params = dict(matchColor = (255,100,30),
			singlePointColor = (255,0,0),
			flags = 0)
	image_matched = scene_box.copy()
	image_matched = cv2.drawMatches(target_gray, keyP, scene_box, keyP2, matches, outImg=image_matched, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
	cv2.imshow("Find Product Live Tester", image_matched)

	# show the frame
	key = cv2.waitKey(1) & 0xFF

	# clear the stream for next frame
	rawCapture.truncate(0)

	# if q is pressed quit loop
	if key == ord("q"):
		break
