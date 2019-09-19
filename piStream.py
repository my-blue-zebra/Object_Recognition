# import packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

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
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF

	# clear the stream for the next frame
	rawCapture.truncate(0)

	# if q is pressed then break from loop
	if key == ord("q"):
		break


