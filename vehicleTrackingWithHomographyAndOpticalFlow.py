# Script to track vehicles with homography and optical flow techniques
# The script will
#   1. Detect cars on the road with YOLOv4 and COCO, and save the
#       bounding boxes
#   2. Use homography to try and find a top-down image of the car, with
#       the bounding box and an estimation of the car's size as a reference
#   3. Use optical flow to track the bounding box through the video, then
#       use the previously found homography to find where the new bounding
#       box is in the top-down image, and estimate how far the car moved
#       in meters.
#   4. Repeat steps 1-2 periodically to find new cars on the road to track

# Ethan Welborn
# ethanwelborn@protonmail.com / ethan.welborn@go.tarleton.edu

import cv2
import numpy as np
import os
import time
import random
import math

### CONFIGURATION SETTINGS

# Video file to read
VIDEO_PATH = "Datasets\\cam_10.mp4"
#VIDEO_PATH = "slow_traffic_small.mp4"

# Do you want the results stored to your computer as a video file? 
# Is overwriting a previous video file okay?
SAVE_RESULT_AS_VIDEO = False
OVERWRITE_PREVIOUS_RESULT = False

# Video file to write to (no effect if SAVE_RESULT_AS_VIDEO = False)
OUTPUT_VIDEO_PATH = "output.mp4"

# What detection model are we using? (YOLO, MASKRCNN)
DETECTION_MODEL = "MASKRCNN"

# What labels are we allowed to track from the COCO dataset?
# This is to ensure we're only tracking vehicles, not other random objects in the frame.
COCO_LABELS_TO_DETECT = ["car", "truck"]

# How confident should our detection model be when detecting cars? (0 is no confidence, 1 is completely confident)
DETECTION_MINIMUM_CONFIDENCE = 0.3

# How many seconds should we wait until the detection model is allowed to detect cars again?
DETECTION_REFRESH_RATE = 2

# Where are the YOLOv4 configuration and weight files contained? (No effect if DETECTION_MODEL != "YOLO")
YOLO_CONFIG_PATH = "yolov4.cfg"
YOLO_WEIGHTS_PATH = "yolov4.weights"

# Where are the Mask-RCNN configuration and weight files contained? (No effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_CONFIG_PATH = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
MASKRCNN_WEIGHTS_PATH = "frozen_inference_graph.pb"

# What threshold value should we use when performing pixelwise segmentation? (Between 0 and 1, no effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_PIXEL_SEGMENTATION_THRESHOLD = 0.2

# Should the detected masks be drawn? If so, what intensity and color should they be drawn with?
# (No effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_DRAW_MASKS = True
MASKRCNN_DRAW_MASKS_INTENSITY = 0.5
# Color is represented in BGR format
MASKRCNN_DRAW_MASKS_COLOR = (255,0,0)

# What method should we use to track bounding boxes in our model?
# Options: naive
BOUNDING_BOX_TRACKING_METHOD = "naive"

# How much overlap is required between the bounding box of a car detected last round versus a car
# detected this round to prove that these two detections are the same car?
MINIMUM_BB_OVERLAP = 0.20

# How fast is the video in frames per second?
FPS = 10

# Should additional information about the algorithm be printed to the console?
DEBUG = True

# Should the time code be printed along with the debug information?
DEBUG_TIMECODE = True

# Should the computer try to automatically detect homographies (False), or should
# the user be prompted to enter the homography manually (True)?
MANUAL_HOMOGRAPHY = True

# Should we draw what the warped image looks like? Has no effect if MANUAL_HOMOGRAPHY is False
DRAW_MANUAL_HOMOGRAPHY = False

# Should the bounding boxes and labels for the detected cars be drawn?
DRAW_BOUNDING_BOXES = True

# How thick should the detection rectangles be? (In pixels)
DRAWING_THICKNESS = 3

# Should the optical flow points be drawn on the video?
DRAW_OPTICAL_FLOW_POINTS = True

# How large should the optical flow points look like when drawn (in pixels)? (No effect if DRAW_OPTICAL_FLOW_POINTS = False)
OPTICAL_FLOW_POINT_DRAWING_SIZE = 5

### END OF CONFIG

# Make sure the detection model is valid
if not(DETECTION_MODEL in ["YOLO","MASKRCNN"]):
    print(DETECTION_MODEL,"is not a valid detection model")
    print("Please try \"YOLO\" or \"MASKRCNN\"")
    exit()

# Make sure the video exists
if not os.path.isfile(VIDEO_PATH):
    print("File at videoPath does not exist:", str(VIDEO_PATH))
    exit()

# Make sure the output video does *not* exist (if applicable)
if SAVE_RESULT_AS_VIDEO and OVERWRITE_PREVIOUS_RESULT == False:
    if os.path.isfile(OUTPUT_VIDEO_PATH):
        print("Output video already exists at the following path:", str(OUTPUT_VIDEO_PATH))
        exit()

# Prepare our detection model and get it loaded in memory
# Give some dummy values just to make sure the variable is in scope - if these
# variables are necessary, then the detection model will fill them in in a minute
maskRCNNModel = 1
YOLOModel = 1
YOLOOutputLayer = 1

if DETECTION_MODEL == "YOLO":
    YOLOModel = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH,YOLO_WEIGHTS_PATH)
    YOLOOutputLayer = [YOLOModel.getLayerNames()[layer - 1] for layer in YOLOModel.getUnconnectedOutLayers()]
elif DETECTION_MODEL == "MASKRCNN":
    maskRCNNModel = cv2.dnn.readNetFromTensorflow(MASKRCNN_WEIGHTS_PATH, MASKRCNN_CONFIG_PATH)

# Set up the video capture
cap = cv2.VideoCapture(VIDEO_PATH)

# Variables and classes for later
COCOLabels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

randomColors = [(255,0,0), (0,255,0), (0,0,255)]

def debugPrint(*args):
    if DEBUG:
        if DEBUG_TIMECODE:
            global totalFrames
            global FPS
            timeSignature = "({seconds}:{frames})".format(seconds=totalFrames//FPS, frames=totalFrames%FPS)
            print(timeSignature,*args)
        else:
            print(*args)

class BoundingBox():
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getCenterX(self):
        return int(self.x + (self.width / 2))

    def getCenterY(self):
        return int(self.y + (self.height / 2))

    def getEndX(self):
        return self.x + self.width

    def getEndY(self):
        return self.y + self.height

    def getOverlap(bb1,bb2):
        # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
        x = max(bb1.getX(), bb2.getX())
        y = max(bb1.getY(), bb2.getY())
        endX = min(bb1.getEndX(), bb2.getEndX())
        endY = min(bb1.getEndY(), bb2.getEndY())

        if endX < x or endY < y:
            return 0.0

        overlapArea = (endX - x) * (endY - y)
        bb1Area = bb1.getWidth() * bb1.getHeight()
        bb2Area = bb2.getWidth() * bb2.getHeight()

        return overlapArea / float(bb1Area + bb2Area - overlapArea)

class Car():
    def __init__(self,ID):
        self.color = random.choice(randomColors)
        self.ID = ID
        self.KMH = 0

currentCars = []
oldCars = []
currentCarID = 0

# Takes in point p in the form of (x, y) and matrix is a 3x3 homography matrix
# https://stackoverflow.com/questions/57399915/how-do-i-determine-the-locations-of-the-points-after-perspective-transform-in-t
def findPointInWarpedImage(p, matrix):
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return (int(px), int(py))

# Only used with manual homography
manualHomographyMatrix = None
manualHomography_pixelsToMeters = 0

# Only used for drawing masks from Mask-RCNN
maskRCNNDrawingMask = None

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# Keep reading frames from the video until it's over
currentTime = time.time()
endTime = time.time()
progress = 0
firstFrame = True
lastGrayFrame = None
totalFrames = 0
while True:
    currentTime = time.time()
    progress += currentTime - endTime
    ret, frame = cap.read()

    totalFrames += 1

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    if SAVE_RESULT_AS_VIDEO and (out is None):
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (frameWidth, frameHeight))

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if MANUAL_HOMOGRAPHY and firstFrame:
        print("Please select the four corners of a car in the frame.")
        print("BLUE: Front left headlight")
        print("RED: Front right headlight")
        print("GREEN: Back left tail-light")
        print("ORANGE: Back right tail-light")
        print("\nWhen you are done, press SPACE to continue.")

        manualHomography_points = [[50,50],[100,50],[50,100],[100,100]]
        manualHomography_selectedPoint = None
        manualHomography_pointSize = 10
        def onMouse(event, x, y, flags, params):
            global manualHomography_selectedPoint
            selectedDistance = 0

            newImage = frame.copy()

            if event == cv2.EVENT_LBUTTONDOWN:
                manualHomography_selectedPoint = None
                for point in manualHomography_points:
                    distance = math.hypot(x - point[0], y - point[1])
                    if math.hypot(x - point[0], y - point[1]) <= manualHomography_pointSize:
                        if manualHomography_selectedPoint and selectedDistance <= distance:
                            pass
                        else:
                            manualHomography_selectedPoint = point
                            selectedDistance = distance
            elif event == cv2.EVENT_LBUTTONUP:
                manualHomography_selectedPoint = None
                selectedDistance = 0

            if manualHomography_selectedPoint != None:
                manualHomography_selectedPoint[0] = x
                manualHomography_selectedPoint[1] = y

            for i in range(len(manualHomography_points)):
                point = manualHomography_points[i]
                color = [(255,100,100),(100,100,255),(100,255,100),(100,175,255)][i]
                cv2.circle(newImage,(point[0], point[1]),manualHomography_pointSize,color,-1)
                cv2.circle(newImage,(point[0], point[1]),manualHomography_pointSize-4,[c/3 for c in color],-1)

            cv2.imshow("Vehicle Tracking", newImage)

        cv2.imshow("Vehicle Tracking", frame)
        cv2.setMouseCallback("Vehicle Tracking", onMouse)
        inputKey = None
        while inputKey != ord(" "):
            inputKey = cv2.waitKey(0) & 0xFF

        # Reset the mouse callback so that it doesn't do anything
        cv2.setMouseCallback("Vehicle Tracking", lambda event, x, y, flags, params: 0)

        # https://www.thezebra.com/resources/driving/average-car-size/
        # Average car length: 14.7 feet
        # Average car width: 5.8 feet
        carReferencePoints = np.array([[0, 0],[58, 0],[0, 147],[58, 147]]) + np.array([[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2]])
        manualHomographyMatrix, status = cv2.findHomography(np.array(manualHomography_points), carReferencePoints)

        #warpedFrame = cv2.warpPerspective(frame, manualHomographyMatrix, (frameWidth*4, frameHeight*4))
        #cv2.imwrite("output.png",warpedFrame)
        #exit()

        print("Homography calculated.")
        meters = float(input("Please estimate the length of the car in meters (from headlight to tail-light): "))
        
        leftHeadLight = findPointInWarpedImage(manualHomography_points[0], manualHomographyMatrix)
        leftTailLight = findPointInWarpedImage(manualHomography_points[2], manualHomographyMatrix)
        manualHomography_pixelsToMeters = math.fabs(leftHeadLight[1] - leftTailLight[1]) / meters

        print("Estimated pixels-to-meters ratio:",manualHomography_pixelsToMeters)

    if firstFrame:
        # Make sure the timer is accurate and ensure that detection is done on the very first frame
        currentTime = time.time()
        progress = DETECTION_REFRESH_RATE

    firstFrame = False

    if ret == False:
        print("Video EOF")
        break

    if progress >= DETECTION_REFRESH_RATE:
        debugPrint(DETECTION_MODEL, "detecting...")

        # Clear the cars list and prepare it for the new cars
        oldCars = currentCars
        currentCars = []

        if DETECTION_MODEL == "YOLO":
            imageBlob = cv2.dnn.blobFromImage(frame, 0.003922, (416,416), swapRB = True, crop = False)
            
            # Propagate and detect objects in the image
            YOLOModel.setInput(imageBlob)
            objectDetectionLayers = YOLOModel.forward(YOLOOutputLayer)

            classIDsList = []
            boxesList = []
            confidencesList = []
            for objectDetectionLayer in objectDetectionLayers:
                for objectDetection in objectDetectionLayer:
                    # Find the most confident prediction for this object
                    allScores = objectDetection[5:]
                    predictedClassID = np.argmax(allScores)
                    predictionConfidence = allScores[predictedClassID]

                    if predictionConfidence >= DETECTION_MINIMUM_CONFIDENCE:
                        predictedClassLabel = COCOLabels[predictedClassID]

                        # We don't care about toothbrushes, hair dryers, or any other random
                        # objects in the COCO dataset - all we want are cars.
                        if not predictedClassLabel in COCO_LABELS_TO_DETECT:
                            continue

                        # Generate the bounding box
                        boundingBox = objectDetection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                        (boxCenterXPoint, boxCenterYPoint, boxWidth, boxHeight) = boundingBox.astype("int")
                        startXPoint = int(boxCenterXPoint - (boxWidth / 2))
                        startYPoint = int(boxCenterYPoint - (boxHeight / 2))

                        classIDsList.append(predictedClassID)
                        confidencesList.append(predictionConfidence)
                        boxesList.append([startXPoint, startYPoint, int(boxWidth), int(boxHeight)])

            # Use non-maximum suppression to avoid generate multiple bounding boxes
            # for the same object
            maxValueIDs = cv2.dnn.NMSBoxes(boxesList, confidencesList, 0.5, 0.4)

            for maxValueID in maxValueIDs:
                box = boxesList[maxValueID]

                predictedClassID = classIDsList[maxValueID]
                predictedClassLabel = COCOLabels[predictedClassID]
                predictionConfidence = confidencesList[maxValueID]

                car = Car(currentCarID)
                car.boundingBox = BoundingBox(box[0],box[1],box[2],box[3])

                # Determine if this car has already been detected, if so, then reuse the old ID
                # to show that this is not a new car.
                bestMatch = None
                bestOverlap = 0
                for otherCar in oldCars:
                    overlap = otherCar.boundingBox.getOverlap(car.boundingBox)
                    if overlap >= MINIMUM_BB_OVERLAP and ((bestMatch is None) or (overlap > bestOverlap)):
                        bestMatch = otherCar
                        bestOverlap = overlap
                #print(bestOverlap, bestMatch.ID if bestMatch is not None else "")

                if bestMatch is None:
                    currentCarID += 1
                else:
                    car.ID = bestMatch.ID
                    car.color = bestMatch.color

                currentCars.append(car)

                # Find points on the car that can be used for optical flow tracking
                mask = np.zeros_like(grayFrame)
                # KEEP IN MIND!! The frame is in the form of [Y][X]
                mask[car.boundingBox.getY():car.boundingBox.getEndY(), car.boundingBox.getX():car.boundingBox.getEndX()] = 255
                car.opticalFlowPoints = cv2.goodFeaturesToTrack(grayFrame, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7)
        
        elif DETECTION_MODEL == "MASKRCNN":
            if MASKRCNN_DRAW_MASKS:
                maskRCNNDrawingMask = np.zeros_like(frame)

            imageBlob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

            # Propagate and detect objects in the image
            maskRCNNModel.setInput(imageBlob)
            boundingBoxes, masks = maskRCNNModel.forward(["detection_out_final", "detection_masks"])

            for i in range(0, boundingBoxes.shape[2]):
                predictedClassID = int(boundingBoxes[0, 0, i, 1])
                predictedClassLabel = COCOLabels[predictedClassID]
                predictionConfidence = boundingBoxes[0, 0, i, 2]

                if predictionConfidence >= DETECTION_MINIMUM_CONFIDENCE and predictedClassLabel in COCO_LABELS_TO_DETECT:
                    car = Car(currentCarID)
                    (x, y, endX, endY) = (boundingBoxes[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])).astype("int")
                    car.boundingBox = BoundingBox(x, y, endX - x, endY - y)

                    # Determine if this car has already been detected, if so, then reuse the old ID
                    # to show that this is not a new car.
                    bestMatch = None
                    bestOverlap = 0
                    for otherCar in oldCars:
                        overlap = otherCar.boundingBox.getOverlap(car.boundingBox)
                        if overlap >= MINIMUM_BB_OVERLAP and ((bestMatch is None) or (overlap > bestOverlap)):
                            bestMatch = otherCar
                            bestOverlap = overlap
                    #print(bestOverlap, bestMatch.ID if bestMatch is not None else "")

                    if bestMatch is None:
                        currentCarID += 1
                    else:
                        car.ID = bestMatch.ID
                        car.color = bestMatch.color

                    currentCars.append(car)

                    # Find the pixel mask of the car
                    mask = masks[i, predictedClassID]
                    mask = cv2.resize(mask, (car.boundingBox.getWidth(), car.boundingBox.getHeight()), interpolation=cv2.INTER_CUBIC)
                    mask = (mask > MASKRCNN_PIXEL_SEGMENTATION_THRESHOLD).astype("uint8") * 255
                    
                    fullImageMask = np.zeros_like(grayFrame)
                    fullImageMask[car.boundingBox.getY():car.boundingBox.getEndY(), car.boundingBox.getX():car.boundingBox.getEndX()] = mask

                    if MASKRCNN_DRAW_MASKS:
                        # Convert the full image mask from greyscale to BGR
                        BGRImageMask = cv2.cvtColor(fullImageMask, cv2.COLOR_GRAY2BGR)
                        
                        # Take the maximum value of the two masks and store the result in the
                        # main drawing mask.
                        maskRCNNDrawingMask = np.maximum.reduce([maskRCNNDrawingMask, (BGRImageMask * (np.array(MASKRCNN_DRAW_MASKS_COLOR) / 255))])

                    #cv2.imshow("Mask", fullImageMask)
                    #cv2.waitKey(2000)

                    # Find corners on the car that are good for tracking
                    car.opticalFlowPoints = cv2.goodFeaturesToTrack(grayFrame, mask=fullImageMask, maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7)
        
        if MASKRCNN_DRAW_MASKS:
            maskRCNNDrawingMask

        debugPrint(DETECTION_MODEL, "detection is complete")
        if MANUAL_HOMOGRAPHY == False:
            # Create homographies for the cars
            debugPrint("Calculating homographies...")
        
            for car in currentCars:
                # VERY PRIMITIVE: We'll use the four corners of the bounding boxes
                # to try and align the car to an estimated 3:1 size (length:width)
                carReferencePoints = np.array([[0, 0],[50, 0],[50, 150],[0, 150]])
                offset = np.array([[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2]])
        
                x = car.boundingBox.getX()
                y = car.boundingBox.getY()
                ex = car.boundingBox.getEndX()
                ey = car.boundingBox.getEndY()

                boundingBoxPoints = np.array([[x,y],[ex,y],[ex,ey],[x,ey]])

                homography, status = cv2.findHomography(boundingBoxPoints, carReferencePoints + offset)

                car.homography = homography

                warpedFrame = cv2.warpPerspective(frame, homography, (frameWidth, frameHeight))
                #cv2.imshow("warpedImage", warpedImage)
                #cv2.waitKey(2000)

            debugPrint("Homographies calculated")

        # Restart the timer.
        currentTime = time.time()
        progress = 0

    # Try to track each car through the video with optical flow
    warpedFrame = cv2.warpPerspective(frame, manualHomographyMatrix, (frameWidth, frameHeight)) if MANUAL_HOMOGRAPHY else 1

    debugPrint("Tracking...")
    if not (lastGrayFrame is None):
        for car in currentCars:
            if car.opticalFlowPoints is not None and len(car.opticalFlowPoints) > 0:
                oldFlowPoints = car.opticalFlowPoints
                newFlowPoints, status, err = cv2.calcOpticalFlowPyrLK(lastGrayFrame, grayFrame, oldFlowPoints, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
                homography = manualHomographyMatrix if MANUAL_HOMOGRAPHY else car.homographyMatrix

                dx = 0
                dy = 0
                if newFlowPoints is not None:
                    # We don't want to keep points that got lost
                    previousLen = len(newFlowPoints)
                    newFlowPoints = newFlowPoints[status == 1].reshape(-1,1,2)
                    if len(newFlowPoints) != previousLen:
                        debugPrint("Lost",previousLen - len(newFlowPoints),"optical flow points")
                        if len(newFlowPoints) == 0:
                            debugPrint("Car completely lost")
                            continue

                    # Save all of our data points so that the next bit of code doesn't remove
                    # points that may become useful next iteration
                    car.opticalFlowPoints = newFlowPoints.reshape(-1,1,2)

                    # Remove outlying data points from our set so that they don't negatively
                    # contribute to our bounding box estimate or our speed estimate
                    
                    if BOUNDING_BOX_TRACKING_METHOD == "naive":
                        minX = newFlowPoints[0][0][0]
                        minY = newFlowPoints[0][0][1]
                        maxX = newFlowPoints[0][0][0]
                        maxY = newFlowPoints[0][0][1]
                        for point in newFlowPoints:
                            if point[0][0] < minX:
                                minX = point[0][0]
                            elif point[0][0] > maxX:
                                maxX = point[0][0]
                            if point[0][1] < minY:
                                minY = point[0][1]
                            elif point[0][1] > maxY:
                                maxY = point[0][1]
                        car.boundingBox = BoundingBox(minX,minY,maxX-minX,maxY-minY)

                    # Estimate the car's speed based on the movement of
                    # the optical flow points in the top-down image
                    for i in range(len(newFlowPoints)):
                        if status[i] == 0:
                            continue
                        oldFlowPoint = findPointInWarpedImage((oldFlowPoints[i][0][0], oldFlowPoints[i][0][1]), homography)
                        newFlowPoint = findPointInWarpedImage((newFlowPoints[i][0][0], newFlowPoints[i][0][1]), homography)
                        dx += newFlowPoint[0] - oldFlowPoint[0]
                        dy += newFlowPoint[1] - oldFlowPoint[1]
                    
                    dx = dx / len(newFlowPoints)
                    dy = dy / len(newFlowPoints)
                    totalMovementInPixels = math.sqrt((dx * dx) + (dy * dy))

                    if MANUAL_HOMOGRAPHY:
                        metersPerSecond = (totalMovementInPixels * FPS) / manualHomography_pixelsToMeters
                        car.KMH = (metersPerSecond * 60 * 60) / 1000
                    else:
                        error("Not yet implemented")

                    
    debugPrint("Tracking complete")

    # Draw the bounding boxes on the screen so that the user can see what's being tracked
    drawingLayer = np.zeros_like(frame)

    if MASKRCNN_DRAW_MASKS and maskRCNNDrawingMask is not None:
        # Add the masks to the frame
        #print(frame.shape,maskRCNNDrawingMask.shape)
        #print(frame.dtype,maskRCNNDrawingMask.dtype)
        frame = cv2.addWeighted(frame, 1 - MASKRCNN_DRAW_MASKS_INTENSITY, maskRCNNDrawingMask.astype("uint8"), MASKRCNN_DRAW_MASKS_INTENSITY, 0)

    for car in currentCars:
        if not (car.opticalFlowPoints is None) and DRAW_OPTICAL_FLOW_POINTS:
            for point in car.opticalFlowPoints:
                transformedPoint = (int(point[0][0]), int(point[0][1]))
                cv2.circle(drawingLayer,transformedPoint,OPTICAL_FLOW_POINT_DRAWING_SIZE,(255,255,255),-1)
                cv2.circle(drawingLayer,transformedPoint,OPTICAL_FLOW_POINT_DRAWING_SIZE-4,(0,0,0),-1)

    if MANUAL_HOMOGRAPHY and DRAW_MANUAL_HOMOGRAPHY:
        # Draw the bounding boxes for the cars on the warped image as well
        for car in currentCars:
            start = findPointInWarpedImage((car.boundingBox.getX(), car.boundingBox.getY()), manualHomographyMatrix)
            end = findPointInWarpedImage((car.boundingBox.getEndX(), car.boundingBox.getEndY()), manualHomographyMatrix)
            minX = min(start[0], end[0])
            minY = min(start[1], end[1])
            maxX = max(start[0], end[0])
            maxY = max(start[1], end[1])
            cv2.rectangle(warpedFrame, (minX, minY), (maxX, maxY), car.color, thickness = DRAWING_THICKNESS)

        for i in range(len(manualHomography_points)):
            point = manualHomography_points[i]
            color = [(255,100,100),(100,100,255),(100,255,100),(100,175,255)][i]
            warpedPoint = findPointInWarpedImage(point, manualHomographyMatrix)
            cv2.circle(warpedFrame,warpedPoint,manualHomography_pointSize,color,-1)
            cv2.circle(warpedFrame,warpedPoint,int(manualHomography_pointSize/2),[c/3 for c in color],-1)

        cv2.imshow("Vehicle Tracking (Bird's Eye)",warpedFrame)

    # Draw the labels last, so that they're on top of everything else in the image
    for car in currentCars:
        if DRAW_BOUNDING_BOXES:
            cv2.rectangle(drawingLayer, (car.boundingBox.getX(), car.boundingBox.getY()), (car.boundingBox.getEndX(), car.boundingBox.getEndY()), car.color, thickness = DRAWING_THICKNESS)
            text = "ID: {ID} ; KMH: {KMH:.2f}".format(ID=car.ID,KMH=car.KMH)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 2
            textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
            #cv2.rectangle(drawingLayer, (car.boundingBox.getX() - textSize[0], car.boundingBox.getY() - textSize[1]), (car.boundingBox.getX(), car.boundingBox.getY()), (255,0,0), thickness = DRAWING_THICKNESS)
            frame[car.boundingBox.getY() - textSize[1]:car.boundingBox.getY(), car.boundingBox.getX():car.boundingBox.getX() + textSize[0]] = (0,0,0)
            drawingLayer[car.boundingBox.getY() - textSize[1]:car.boundingBox.getY(), car.boundingBox.getX():car.boundingBox.getX() + textSize[0]] = (0,0,0)
            cv2.putText(drawingLayer, text, (car.boundingBox.getX(), car.boundingBox.getY()), fontFace, fontScale, (255,255,255), thickness)  

    frame = cv2.add(frame,drawingLayer)

    # Save all results to video if applicable
    if SAVE_RESULT_AS_VIDEO:
        out.write(frame)

    cv2.imshow("Vehicle Tracking", frame)
    inputKey = cv2.waitKey(int((1 / FPS) * 1000)) & 0xFF
    if inputKey == 27 or inputKey == ord("q") or inputKey == ord("Q"): # Esc or Q to quit
        break

    # Save the current gray frame for next iteration
    lastGrayFrame = grayFrame
    # Prepare the timer for next iteration
    endTime = currentTime

cap.release()
if SAVE_RESULT_AS_VIDEO:
    out.release()
cv2.destroyAllWindows()
