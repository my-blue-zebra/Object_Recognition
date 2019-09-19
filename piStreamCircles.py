# import packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
## import sys

# init the camera and grab a reference
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = camera.resolution)

# allow camera to warmup
time.sleep(0.1)

# camera frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw array of the image
	image = frame.array

	# show the frame
	# cv2.imshow("Frame", image)
	# key = cv2.waitKey(1) & 0xFF

	# Analyse picture for any circles
	frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	frame_blur = cv2.medianBlur(frame_gray, 5)
	circles = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 1, 50)
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			cv2.circle(image, (i[0],i[1]), i[2], (0,255,0), 2)
		##sys.stdout.write((circles[:,0],circles[:,1]))
	# show the frame
	cv2.imshow("Live Circle Finder", image)
	key = cv2.waitKey(1) & 0xFF

	# clear the stream for the next frame
	rawCapture.truncate(0)

	# if q is pressed then break from loop
	if key == ord("q"):
		break


