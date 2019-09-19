# Chocolate bar finder
# Original Author: Mike Talbot
# Date of Creation: 03/09/19

# Add steps as they are completed - keep track of what is done
# Step 1 take in arguements
# Step 2 find keypoints of products
# Step 3 take a frame of the live stream
# Step 4 find all contours of a good size
# Step 5 highlight contours
# Step 6 isolate good contours
# Step 7 use ORB detection with Brute Force matching

##### IMPORTS #####
from scipy.spatial import distance as dist
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import time
import cv2
import os

##### FUNCTIONS #####

# Contour Finder - Selects products in the image
def Contours(gray_image):
    temp = cv2.Canny(gray_image, 50, 100)
    temp = cv2.dilate(temp, None, iterations=1)
    temp = cv2.erode(temp, None, iterations=1)
    cv2.imshow("EdgeDETECT", temp)
    cnts = cv2.findContours(temp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    return cnts

# Crop Image and reveal only the product
def Take(original_image,pts,label=None):
    # this function is only designed for rectangle and not any four sided shape
    (tl, tr, br, bl) = pts #May need to order the points before hand -> look into after
    Width = int(dist.euclidean(tl,tr))
    Height = int(dist.euclidean(tl,bl))
    dst = np.array([[0,0], [Width-1,0], [Width-1,Height-1], [0,Height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    focused = cv2.warpPerspective(original_image, M , (Width, Height))
    if label is not None:
	Label = str(label)
	cv2.imshow(Label, focused)
    return(focused)

# Uses a keypoint detector (any work as long as they have an inbuilt  detectAndCompute function) 
# Then runs through to see if the product matches with a label from the set
def PatternMatcher(test_image, list_kpd, kpgen, matcher):
    test_gray=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    kpl,dl =kpgen.detectAndCompute(test_gray, None)
    NumMatches = []
    matchList = []
    for KPD in list_kpd:
	kpp,dp,im = KPD
    	try:
	    matches = matcher.match(dp, dl)
    	    matches = sorted(matches, key = lambda x:x.distance)
	    matchList.append(matches)
	    NumMatches.append(len(matches))
	except:
	    NumMatches.append(0)
    if NumMatches == [0]*len(list_kpd):
   	print("No Match")
    else:
	Bestmatch = NumMatches.index(max(NumMatches))
    	# Print labels of products
    	if Bestmatch == 0:
	    print("Cinnamon Spice")
    	elif Bestmatch == 1:
	    print("Cranberry & Raspberry")
    	elif Bestmatch == 2:
	    print("Coffee & Cardamom")
	imgMatch = test_image.copy()
	imgMatch = cv2.drawMatches(list_kpd[Bestmatch][2], list_kpd[Bestmatch][0],test_image, kpl, matchList[Bestmatch], outImg = imgMatch, flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
	cv2.imshow("Test what's Matching", imgMatch)
	return list_kpd[Bestmatch]


##### START OF METHOD #####

# Arguement Parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--products", required = True, help = "path to directory holding all reference images of your products")
ap.add_argument("-w", "--ref-width", required = True, help = "width of the size reference image in the top left of frame" )
ap.add_argument("-ew", "--expected-width", required = True, help = "expected width of product")
ap.add_argument("-eh", "--expected-height", required = True, help = "expected height of product")
ap.add_argument("-t", "--tolerance", default = 0.05, help = "tolerance level to counteract any inaccuracies from finding contours")

args = vars(ap.parse_args())

# Detect Keypoints in the the reference images
listOfKeyPointsAndDescriptors = []
orb = cv2.ORB_create(nfeatures=5000, scoreType = cv2.ORB_FAST_SCORE)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck = True)
for image_path in os.listdir(args["products"]):
    full_path = os.path.join(args["products"],image_path)

    try:
        product_image = cv2.imread(full_path)
    except:
        print(full_path + " is not a readable file")
	continue
    product_image = cv2.resize(product_image, None,  fx=0.25, fy=0.25)
    product_gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
    kp, d =orb.detectAndCompute(product_gray, None)
    KPD = (kp,d,product_image)
    listOfKeyPointsAndDescriptors.append(KPD)
    print("Key Points for: " + image_path + " Complete")
    cv2.imshow(image_path,product_image)

# Initialise the camera for the stream
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = camera.resolution)
print("Started Capture")
time.sleep(1)

# Take apart stream frame by frame
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port = True):
    # grab the raw image
    scene = frame.array
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.medianBlur(scene_gray,5)
    orig_frame = scene.copy()
    # find contours of frame
    refObj = None
    cnts = Contours(scene_gray)
    listOfBoxes = []
    if cnts is None: # skips to next frame until there is contours
	cv2.imshow("ContourGetter", orig_frame)
	key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)
	if key == ord("q"):break
	continue

    i = 0 #Counter for contour pictures

    for c in cnts:
	if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 15000: # gets rid of small contours
            #cv2.imshow("ContourGetter", orig_frame)
	    key = cv2.waitKey(1) & 0xFF
	    rawCapture.truncate(0)
   	    if key == ord("q"):break
	    continue
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        listOfBoxes.append(box)
        # Use Reference object to get scaling
        if refObj == None:
            left = (box[0,:] + box[3,:])/2
            right = (box[1,:] + box[2,:])/2
            D = dist.euclidean(left,right)
            convD = D/float(args["ref_width"])
            cv2.drawContours(orig_frame, [box.astype("int")], -1, (255,0,255), 5)
            refObj = (box, convD)

	# Measure rest of contours to get the correct size ones
	ExpDim1 = float(args["expected_height"])
	ExpDim2 = float(args["expected_width"])
	tol = float(args["tolerance"])

	D1 = dist.euclidean(box[0,:],box[1,:])/convD
	D2 = dist.euclidean(box[1,:],box[2,:])/convD
	snapshot = False # determines whether a picture should be taken
	if ((ExpDim1*(1-tol) <= D1) and (D1 <= ExpDim1*(1+tol))and((ExpDim2*(1-tol) <= D2) and (D2 <= ExpDim2*(1+tol)))):
        # Draw out contours to see if it is detecting the products
            cv2.drawContours(orig_frame, [box.astype("int")], -1, (0,255,0), 2)
	    snapshot = True
	    i += 1
	elif ((ExpDim1*(1-tol) <= D2) and (D2 <= ExpDim1*(1+tol))and((ExpDim2*(1-tol) <= D1) and (D1 <= ExpDim2*(1+tol)))):
            cv2.drawContours(orig_frame, [box.astype("int")], -1, (255,255,0), 2)
 	    snapshot = True
	    i += 1
	else:
	    cv2.drawContours(orig_frame, [box.astype("int")], -1, (0,0,255), 1)
	    cx, cy = (np.average(box[:,0]),np.average(box[:,1]))
	    cv2.putText(orig_frame, "{:2.2f}mm, {:2.2f}".format(D1,D2), (int(cx),int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
	if snapshot:
	    cropped = Take(scene,box,i)
	    #(kpm,dm,im) = 
	    PatternMatcher(cropped, listOfKeyPointsAndDescriptors, orb, bf)

    cv2.imshow("ContourGetter", orig_frame)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):break
