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
# Step 7 remove and square the logos
# Step 8 use HOG with K-NN (k=1)

##### IMPORTS #####
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance as dist
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils import perspective
from imutils import contours
from imutils import paths # may not need due to os
from skimage import exposure
from skimage import feature
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

##### FUNCTIONS #####

# Contour Finder - Selects products in the image
def Contours(gray_image, print_true = True, test = False):
    #temp = cv2.Canny(gray_image, 50, 100)
    temp = imutils.auto_canny(gray_image)
    if test: cv2.imshow("Canny A", temp)
    else:
    	temp = cv2.dilate(temp, None, iterations=1)
    	temp = cv2.erode(temp, None, iterations=1)
    if test: cv2.imshow("Canny B", temp)
    if(print_true): cv2.imshow("EdgeDETECT", temp)
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

# Crop the iomage further to remove the label
def TakeLabel(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cnts = Contours(gray,False, True)
    for c in cnts:
    	pts = cv2.minAreaRect(cnts[0]) # as list is sorted with largest first
    	box = cv2.cv.BoxPoints(pts) if imutils.is_cv2() else cv2.boxPoints(pts)
    	box = np.array(box, dtype="int")
    	box = perspective.order_points(box)
    	(tl, tr, br, bl) = box #May need to order the points before hand -> look into after
    	Width = int(dist.euclidean(tl,tr))
    	Height = int(dist.euclidean(tl,bl))
    	dst = np.array([[0,0], [Width-1,0], [Width-1,Height-1], [0,Height-1]], dtype="float32")
    	cropped = cv2.drawContours(cropped, [box.astype("int")], -1, (255,255,0), 2)
    #cv2.imshow("CONTOURS",cropped)
    M = cv2.getPerspectiveTransform(box, dst)
    focused = cv2.warpPerspective(cropped, M , (Width, Height))
    cv2.imshow("LABEL",focused)

# using the model generated in order to predict what label is fed into the predictor
def Predict(image,model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = imutils.auto_canny(gray)
    resized = cv2.resize(gray,image_resize)
    H, hogImage = feature.hog(resized, orientations=9, pixels_per_cell=ppc,
	 cells_per_block=cpb, transform_sqrt=True, block_norm="L1", visualize=True)
    pred = model.predict(H.reshape(1,-1))[0]
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0,255))
    hogImage = hogImage.astype("uint8")
    #cv2.imshow("Hog Image No. {}".format(i+1), hogImage)
    cv2.putText(image, pred.title(), (10,35), cv2.FONT_HERSHEY_SIMPLEX,1.0,(110,34,59),3)
    cv2.imshow("Test Image No. {}".format(i+1), image)

##### START OF METHOD #####

# HOG vars
ppc = (4,4)
cpb = (2,2)
image_resize = (100,200)

# Arguement Parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--products", required = True, help = "path to directory holding all reference images of your products")
ap.add_argument("-w", "--ref-width", required = True, help = "width of the size reference image in the top left of frame" )
ap.add_argument("-ew", "--expected-width", required = True, help = "expected width of product")
ap.add_argument("-eh", "--expected-height", required = True, help = "expected height of product")
ap.add_argument("-t", "--tolerance", default = 0.05, help = "tolerance level to counteract any inaccuracies from finding contours")
ap.add_argument("-m", "--model", default = None, help = "path to file containing HOG model for KNN classification")

args = vars(ap.parse_args())

# Create the model for 1-NN
if args["model"] is None:
    print("[INFO] Extracting features...")
    data = []
    labels = []
    for image_path in paths.list_images(args["products"]):
        print("[INFO] " + image_path + " loaded.")
        flavour = image_path.split("/")[-2]
        try:
            product_image = cv2.imread(image_path)
        except:
            print(full_path + " is not a readable file")
            continue
        product_gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(product_gray)
        logo = cv2.resize(edged, image_resize)
        for angle in np.arange(0,360,90):
            logo_rot = imutils.rotate_bound(logo, angle)
            H,hogImage = feature.hog(logo_rot, orientations = 9,  pixels_per_cell=ppc,
	         cells_per_block=cpb, transform_sqrt=True, block_norm="L1", visualize=True)
            hogImage = exposure.rescale_intensity(hogImage, out_range=(0,255))
            hogImage = hogImage.astype("uint8")
            #cv2.imshow(flavour,product_image)
            #cv2.imshow("HOG class "+image_path , hogImage)
            data.append(H)
            labels.append(flavour)

    # Train Classifiers
    print("[INFO] Training classifier...")
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(data, labels)
    print("[INFO] Creating model file...")
    pickle.dump(model, open("modelFile.txt", 'wb'))
else:
    print("[INFO] Obtaining model...")
    model = pickle.load(open(args["model"], 'rb'))


# Initialise the camera for the stream
camera = PiCamera()
camera.resolution = (960,720)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = camera.resolution)
print("[INFO] Starting Capture...")
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
        if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 60000: # gets rid of small contours
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
	    cropped = Take(scene,box)
	    #label = TakeLabel(cropped) matchging with full images
	    Predict(cropped,model)

    cv2.imshow("ContourGetter", orig_frame)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q") or key == ord("Q"):break
print("[INFO] Closing program.")
