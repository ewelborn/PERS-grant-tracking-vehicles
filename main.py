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
import config

# Ensure we're using the proper OpenCV package
if hasattr(cv2, "legacy") == False:
    print("cv2.legacy is missing! Please install (or enable) an OpenCV version that has this package.")
    print("The legacy package is required so that the multitracker object can be used.")
    print("Recommended pip package is 'opencv-contrib-python' (4.6.0.66)")

# Make sure the detection model is valid
config.DETECTION_MODEL = config.DETECTION_MODEL.upper()
if not(config.DETECTION_MODEL in ["YOLO", "MASKRCNN"]):
    print(config.DETECTION_MODEL, "is not a valid detection model")
    print("Please try \"YOLO\" or \"MASKRCNN\"")
    exit()

# Make sure that HOMOGRAPHY_INFERENCE is valid
config.HOMOGRAPHY_INFERENCE = config.HOMOGRAPHY_INFERENCE.upper()
if not (config.HOMOGRAPHY_INFERENCE in ["MANUAL", "AUTO", "LOADFILE"]):
    print(config.HOMOGRAPHY_INFERENCE, "is not a valid inference setting")
    print("Please try \"MANUAL\", \"AUTO\", or \"LOADFILE\"")
    exit()

# If we're loading the homography from a file, ensure that it's loading correctly
manualHomographyMatrix = None
if config.HOMOGRAPHY_INFERENCE == "LOADFILE":
    if not os.path.isfile(config.HOMOGRAPHY_FILE):
        print("Could not find homography file:",config.HOMOGRAPHY_FILE)
        exit()

    homographyFile = np.load(config.HOMOGRAPHY_FILE)
    manualHomographyMatrix = homographyFile["manualHomographyMatrix"]
    config.HOMOGRAPHY_SCALING_FACTOR = homographyFile["homographyScalingFactor"]

# If we're saving the homography to a file, make sure it doesn't already exist!
# (Or if it does, make sure we're allowed to overwrite it)
if config.HOMOGRAPHY_INFERENCE == "MANUAL" and config.HOMOGRAPHY_SAVE_TO_FILE:
    if config.HOMOGRAPHY_SAVE_TO_FILE_OVERWRITE == False and os.path.isfile(config.HOMOGRAPHY_FILE):
        print("Manual homography already exists at path:",config.HOMOGRAPHY_FILE)
        exit()

# Make sure the tracking model is valid
trackingModels = {
    "BOOSTING": lambda: cv2.legacy.TrackerBoosting_create(),
    "MIL": lambda: cv2.legacy.TrackerMIL_create(),
    "KCF": lambda: cv2.legacy.TrackerKCF_create(),
    "TLD": lambda: cv2.legacy.TrackerTLD_create(),
    "MEDIANFLOW": lambda: cv2.legacy.TrackerMedianFlow_create(),
    "GOTURN": lambda: cv2.legacy.TrackerGOTURN_create(),
    "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create(),
    "CSRT": lambda: cv2.legacy.TrackerCSRT_create(),
}
config.BOUNDING_BOX_TRACKING_MODEL = config.BOUNDING_BOX_TRACKING_MODEL.upper()
if not (config.BOUNDING_BOX_TRACKING_MODEL in trackingModels.keys()):
    print(config.BOUNDING_BOX_TRACKING_MODEL,"is not a valid tracking model")
    print("Please try any of the following:", " ".join(trackingModels.keys()))
    exit()

# Make sure the video exists
if not os.path.isfile(config.VIDEO_PATH):
    print("File at videoPath does not exist:", str(config.VIDEO_PATH))
    exit()

# Make sure the output video does *not* exist (if applicable)
if config.SAVE_RESULT_AS_VIDEO and config.OVERWRITE_PREVIOUS_RESULT == False:
    if os.path.isfile(config.OUTPUT_VIDEO_PATH):
        print("Output video already exists at the following path:", str(config.OUTPUT_VIDEO_PATH))
        exit()

# Prepare our detection model and get it loaded in memory
# Give some dummy values just to make sure the variable is in scope - if these
# variables are necessary, then the detection model will fill them in in a minute
maskRCNNModel = 1
YOLOModel = 1
YOLOOutputLayer = 1

if config.DETECTION_MODEL == "YOLO":
    YOLOModel = cv2.dnn.readNetFromDarknet(config.YOLO_CONFIG_PATH, config.YOLO_WEIGHTS_PATH)
    YOLOOutputLayer = [YOLOModel.getLayerNames()[layer - 1] for layer in YOLOModel.getUnconnectedOutLayers()]
elif config.DETECTION_MODEL == "MASKRCNN":
    maskRCNNModel = cv2.dnn.readNetFromTensorflow(config.MASKRCNN_WEIGHTS_PATH, config.MASKRCNN_CONFIG_PATH)

# Set up the video capture
cap = cv2.VideoCapture(config.VIDEO_PATH)

# Variables and classes for later
COCOLabels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

randomColors = [(255,0,0), (0,255,0), (0,0,255)]

# back-left, back-right, front-right, front-left
# https://www.thezebra.com/resources/driving/average-car-size/
# Average car length: 14.7 feet --> 4.48 meters
# Average car width: 5.8 feet --> 1.77 meters
MEAN_SEDAN_BOUNDING_BOX = [[0, 0],[1.77 * config.HOMOGRAPHY_SCALING_FACTOR, 0],[1.77 * config.HOMOGRAPHY_SCALING_FACTOR, 4.48 * config.HOMOGRAPHY_SCALING_FACTOR],[0, 4.48 * config.HOMOGRAPHY_SCALING_FACTOR]]

def debugPrint(*args):
    if config.DEBUG:
        if config.DEBUG_TIMECODE:
            global totalFrames
            timeSignature = "({seconds}:{frames})".format(seconds=totalFrames//config.FPS, frames=totalFrames%config.FPS)
            print(timeSignature,*args)
        else:
            print(*args)

# 2D
class BoundingBox():
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def __str__(self):
        return "[{x}, {y}, {width}, {height}]".format(x=self.x, y=self.y, width=self.width, height=self.height)

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

    def getArea(self):
        return self.width * self.height

    def getOverlap(bb1,bb2):
        # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
        x = max(bb1.getX(), bb2.getX())
        y = max(bb1.getY(), bb2.getY())
        endX = min(bb1.getEndX(), bb2.getEndX())
        endY = min(bb1.getEndY(), bb2.getEndY())

        if endX < x or endY < y:
            return 0.0

        overlapArea = (endX - x) * (endY - y)

        return overlapArea / float(bb1.getArea() + bb2.getArea() - overlapArea)

class Car():
    def __init__(self,ID):
        self.color = random.choice(randomColors)
        self.ID = ID
        self.recordedKMH = []
        self.screenPoints = []
        self.u_screenPoints = []
        self.previousTheta = 0
        self.height = 0

    def recordSpeed(self, KMH):
        self.recordedKMH.append(KMH)
        if len(self.recordedKMH) > config.VEHICLE_SPEED_ESTIMATION_SMOOTHING_FRAMES:
            self.recordedKMH.pop(0)

    def getKMH(self):
        # Return the average of the recorded speeds, or 0 if there are no recorded speeds
        return (sum(self.recordedKMH) / len(self.recordedKMH)) if len(self.recordedKMH) > 0 else 0

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
manualHomography_pixelsToMeters = 0
manualHomography_pointSize = 10

multiTracker = cv2.legacy.MultiTracker_create()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# Keep reading frames from the video until it's over
progress = 0
firstFrame = True
lastGrayFrame = None
detectedThisFrame = False
totalFrames = 0
while True:
    progress += 1
    ret, frame = cap.read()

    # Keep running until we've processed all of the frames that the user requested,
    # or we reached the end of the video.
    if (ret == False) or ((totalFrames >= config.MAX_FRAMES_TO_PROCESS) and config.MAX_FRAMES_TO_PROCESS > -1):
        break

    totalFrames += 1

    detectedThisFrame = False

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    if config.SAVE_RESULT_AS_VIDEO and (out is None):
        out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, config.FPS, (frameWidth, frameHeight))

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if config.HOMOGRAPHY_INFERENCE == "MANUAL" and firstFrame:
        if config.HOMOGRAPHY_SAVE_TO_FILE:
            print("The following selection will be saved to disk at", config.HOMOGRAPHY_FILE, "\n")

        print("Please select the four corners of a car in the frame.")
        print("BLUE: Front left headlight")
        print("RED: Front right headlight")
        print("GREEN: Back left tail-light")
        print("ORANGE: Back right tail-light")
        print("\nWhen you are done, press SPACE to continue.")

        manualHomography_points = [[50,50],[100,50],[100,100],[50,100]]
        manualHomography_selectedPoint = None
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
                color = [(255,100,100),(100,100,255),(100,175,255),(100,255,100)][i]
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
        carReferencePoints = np.array(MEAN_SEDAN_BOUNDING_BOX) + np.array([[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2]])
        manualHomographyMatrix, status = cv2.findHomography(np.array(manualHomography_points), carReferencePoints)

        print("Homography calculated.")

        # Save the homography matrix if applicable
        if config.HOMOGRAPHY_SAVE_TO_FILE:
            np.savez(config.HOMOGRAPHY_FILE, manualHomographyMatrix=manualHomographyMatrix, homographyScalingFactor=config.HOMOGRAPHY_SCALING_FACTOR)
            print("Homography saved.")

        #meters = float(input("Please estimate the length of the car in meters (from headlight to tail-light): "))
        
        #leftHeadLight = findPointInWarpedImage(manualHomography_points[0], manualHomographyMatrix)
        #leftTailLight = findPointInWarpedImage(manualHomography_points[2], manualHomographyMatrix)
        #manualHomography_pixelsToMeters = math.fabs(leftHeadLight[1] - leftTailLight[1]) / meters

        #print("Estimated pixels-to-meters ratio:",manualHomography_pixelsToMeters)

    if firstFrame:
        # Make sure the timer is accurate and ensure that detection is done on the very first frame
        progress = config.DETECTION_REFRESH_RATE

    firstFrame = False

    if ret == False:
        print("Video EOF")
        break

    if progress >= config.DETECTION_REFRESH_RATE:
        debugPrint(config.DETECTION_MODEL, "detecting...")
        detectedThisFrame = True

        # Clear the cars list and prepare it for the new cars
        oldCars = currentCars
        currentCars = []

        # Clear the multitracker as well
        multiTracker = cv2.legacy.MultiTracker_create()

        if config.DETECTION_MODEL == "YOLO":
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

                    if predictionConfidence >= config.DETECTION_MINIMUM_CONFIDENCE:
                        predictedClassLabel = COCOLabels[predictedClassID]

                        # We don't care about toothbrushes, hair dryers, or any other random
                        # objects in the COCO dataset - all we want are cars.
                        if not predictedClassLabel in config.COCO_LABELS_TO_DETECT:
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
                    if overlap >= config.MINIMUM_BB_OVERLAP and ((bestMatch is None) or (overlap > bestOverlap)):
                        bestMatch = otherCar
                        bestOverlap = overlap
                #print(bestOverlap, bestMatch.ID if bestMatch is not None else "")

                if bestMatch is None:
                    currentCarID += 1
                else:
                    car.ID = bestMatch.ID
                    car.color = bestMatch.color
                    car.recordedKMH = bestMatch.recordedKMH

                currentCars.append(car)

                # Find points on the car that can be used for optical flow tracking
                mask = np.zeros_like(grayFrame)
                # KEEP IN MIND!! The frame is in the form of [Y][X]
                mask[car.boundingBox.getY():car.boundingBox.getEndY(), car.boundingBox.getX():car.boundingBox.getEndX()] = 255
                
                # Register the car's bounding box with the multitracker
                # Use the tracking model that the user specified in the config settings
                multiTracker.add(trackingModels.get(config.BOUNDING_BOX_TRACKING_MODEL)(), frame, [car.boundingBox.getX(), car.boundingBox.getY(), car.boundingBox.getWidth(), car.boundingBox.getHeight()])
        
        elif config.DETECTION_MODEL == "MASKRCNN":
            imageBlob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

            # Propagate and detect objects in the image
            maskRCNNModel.setInput(imageBlob)
            boundingBoxes, masks = maskRCNNModel.forward(["detection_out_final", "detection_masks"])

            # Use non-maximum suppression to avoid generate multiple bounding boxes
            # for the same object
            boxesList = [boundingBoxes[0, 0, i, 3:7] for i in range(0, boundingBoxes.shape[2])]
            confidencesList = [boundingBoxes[0, 0, i, 2] for i in range(0, boundingBoxes.shape[2])]

            # NMSBoxes takes two special arguments - score_threshold and nms_threshold
            # score_threshold is used to filter out low confidence results (i.e. it's the minimum
            #   confidence necessary to keep a result)
            # nms_threshold is the maximum intersection allowed between two results
            maxValueIDs = cv2.dnn.NMSBoxes(boxesList, confidencesList, config.DETECTION_MINIMUM_CONFIDENCE, config.NMS_THRESHOLD)

            for i in maxValueIDs:
                predictedClassID = int(boundingBoxes[0, 0, i, 1])
                predictedClassLabel = COCOLabels[predictedClassID] if predictedClassID < len(COCOLabels) else ""
                predictionConfidence = boundingBoxes[0, 0, i, 2]

                if predictedClassLabel in config.COCO_LABELS_TO_DETECT:
                    car = Car(currentCarID)
                    x, y, endX, endY = (boundingBoxes[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])).astype("int")
                    car.boundingBox = BoundingBox(x, y, endX - x, endY - y)

                    # Determine if this car has already been detected, if so, then reuse the old ID
                    # to show that this is not a new car.
                    bestMatch = None
                    bestOverlap = 0
                    for otherCar in oldCars:
                        overlap = otherCar.boundingBox.getOverlap(car.boundingBox)
                        if overlap >= config.MINIMUM_BB_OVERLAP and ((bestMatch is None) or (overlap > bestOverlap)):
                            bestMatch = otherCar
                            bestOverlap = overlap
                    #print(bestOverlap, bestMatch.ID if bestMatch is not None else "")

                    if bestMatch is None:
                        currentCarID += 1
                    else:
                        car.ID = bestMatch.ID
                        car.color = bestMatch.color
                        car.recordedKMH = bestMatch.recordedKMH
                        car.screenPoints = bestMatch.screenPoints

                    currentCars.append(car)

                    # Find the pixel mask of the car
                    mask = masks[i, predictedClassID]
                    mask = cv2.resize(mask, (car.boundingBox.getWidth(), car.boundingBox.getHeight()), interpolation=cv2.INTER_CUBIC)
                    mask = (mask > config.MASKRCNN_PIXEL_SEGMENTATION_THRESHOLD).astype("uint8") * 255
                    car.mask = mask

                    # Register the car's bounding box with the multitracker
                    # Use the tracking model that the user specified in the config settings
                    multiTracker.add(trackingModels.get(config.BOUNDING_BOX_TRACKING_MODEL)(), frame, [x, y, endX - x, endY - y])

            # Last step: update the multitracker with the current frame so that it has context.
            # This prevents the bug where the first tracking frame after a detection has all
            # of the vehicles reporting 0 for their initial speed.
            multiTracker.update(frame)

        debugPrint(config.DETECTION_MODEL, "detection is complete")
        if config.HOMOGRAPHY_INFERENCE == "AUTO":
            # Create homographies for the cars
            debugPrint("Calculating homographies...")
        
            for car in currentCars:
                # VERY PRIMITIVE: We'll use the four corners of the bounding boxes
                # to try and align the car to a mean 3:1 size (length:width)
                # https://www.thezebra.com/resources/driving/average-car-size/
                # Average car length: 14.7 feet --> 4.48 meters
                # Average car width: 5.8 feet --> 1.77 meters
                carReferencePoints = MEAN_SEDAN_BOUNDING_BOX
                offset = np.array([[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2]])
                
                # Naive: use the bounding box coordinates for the homography
                x = car.boundingBox.getX()
                y = car.boundingBox.getY()
                ex = car.boundingBox.getEndX()
                ey = car.boundingBox.getEndY()
                if config.DETECTION_MODEL == "MASKRCNN":
                    # We have the mask of the car, use that instead
                    pass
                    #corners = cv2.goodFeaturesToTrack(car.mask, maxCorners=4, qualityLevel=0.3, minDistance=3, blockSize=7)
                    #print(corners)
                    #newMask = car.mask.copy()
                    #newMask = cv2.cvtColor(newMask, cv2.COLOR_GRAY2BGR)
                    #for corner in corners:
                    #    print([int(a) for a in corner[0]])
                    #    cv2.circle(newMask,[int(a) for a in corner[0]],3,(255,0,0),-1)
                    #cv2.imshow("Mask",newMask)
                    #cv2.waitKey(0)

                boundingBoxPoints = np.array([[x,y],[ex,y],[ex,ey],[x,ey]])

                homography, status = cv2.findHomography(boundingBoxPoints, carReferencePoints + offset)

                car.homographyMatrix = homography

                warpedFrame = cv2.warpPerspective(frame, homography, (frameWidth, frameHeight))
                #cv2.imshow("warpedImage", warpedImage)
                #cv2.waitKey(2000)

            #cap.release()
            #cv2.destroyAllWindows()
            #exit()
            debugPrint("Homographies calculated")

        # Restart the timer.
        progress = 0

    # Try to track each car through the video with optical flow
    warpedFrame = cv2.warpPerspective(frame, manualHomographyMatrix, (frameWidth, frameHeight)) if config.HOMOGRAPHY_INFERENCE != "AUTO" else 1

    if (not (lastGrayFrame is None)) and detectedThisFrame == False:
        debugPrint("Tracking with {model}...".format(model=config.BOUNDING_BOX_TRACKING_MODEL))
        success, boxes = multiTracker.update(frame)

        for i in range(len(currentCars)):
            car = currentCars[i]
            box = boxes[i]
            #print([car.boundingBox.getX(),car.boundingBox.getY(),car.boundingBox.getWidth(),car.boundingBox.getHeight()],"vs",box)
            
            oldBoundingBox = car.boundingBox
            car.boundingBox = BoundingBox(box[0], box[1], box[2], box[3])
            if car.boundingBox.getArea() != 0 and oldBoundingBox is not None:
                homography = manualHomographyMatrix if config.HOMOGRAPHY_INFERENCE != "AUTO" else car.homographyMatrix

                # Estimate the car's speed based on the movement of
                # the bounding box in the top-down image
                oldX, oldY = findPointInWarpedImage((oldBoundingBox.getCenterX(), oldBoundingBox.getCenterY()), homography)
                newX, newY = findPointInWarpedImage((car.boundingBox.getCenterX(), car.boundingBox.getCenterY()), homography)
                dx = newX - oldX
                dy = newY - oldY

                # Generate the 3D bounding box
                # Find the angle of motion
                thetaInRadians = math.atan2(dy, dx) + (math.pi / 2)
                # If we don't have any information to adjust the angle of motion
                # (the car didn't move), then use the last angle of motion instead
                if dx == 0 and dy == 0:
                    thetaInRadians = car.previousTheta
                else:
                    car.previousTheta = thetaInRadians
                #print(dx,dy,thetaInRadians)
                carReferencePoints = np.array(MEAN_SEDAN_BOUNDING_BOX)

                # Determine if the 3D bounding box is properly fitting the vehicle or not
                boundingBoxPoints = (
                    (car.boundingBox.getX(),    car.boundingBox.getY()),
                    (car.boundingBox.getX(),    car.boundingBox.getEndY()),
                    (car.boundingBox.getEndX(), car.boundingBox.getEndY()),
                    (car.boundingBox.getEndX(), car.boundingBox.getY()),
                )
                boundingBoxPoints = [findPointInWarpedImage(point, homography) for point in boundingBoxPoints]
                
                boundingBoxMinX = min([point[0] for point in boundingBoxPoints])
                boundingBoxMinY = min([point[1] for point in boundingBoxPoints])
                boundingBoxMaxX = max([point[0] for point in boundingBoxPoints])
                boundingBoxMaxY = max([point[1] for point in boundingBoxPoints])
                
                boundingBoxWidth = boundingBoxMaxX - boundingBoxMinX
                boundingBoxHeight = boundingBoxMaxY - boundingBoxMinY

                referenceMinX = min([point[0] for point in carReferencePoints])
                referenceMinY = min([point[1] for point in carReferencePoints])
                referenceMaxX = max([point[0] for point in carReferencePoints])
                referenceMaxY = max([point[1] for point in carReferencePoints])
                
                referenceWidth = referenceMaxX - referenceMinX
                referenceHeight = referenceMaxY - referenceMinY

                scale = min(boundingBoxWidth / referenceWidth, boundingBoxHeight / referenceHeight) * 0.8

                carReferencePoints = carReferencePoints * scale

                car.height = int(referenceHeight * 0.6)

                # Translate the reference points so that the center of the
                # rectangle is 0,0
                centerX = max([p[0] for p in carReferencePoints]) / 2
                centerY = max([p[1] for p in carReferencePoints]) / 2
                carReferencePoints -= np.array([[centerX, centerY],[centerX, centerY],[centerX, centerY],[centerX, centerY]])
                #print(carReferencePoints)

                # Rotate the reference points around the origin (0,0)
                for point in carReferencePoints:
                    # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
                    px = (math.cos(thetaInRadians) * (point[0])) - (math.sin(thetaInRadians) * (point[1]))
                    py = (math.sin(thetaInRadians) * (point[0])) + (math.cos(thetaInRadians) * (point[1]))
                    point[0] = px
                    point[1] = py

                # Translate the reference points so that their new origin is the center
                # of the car's new bounding box
                carReferencePoints += np.array([[newX, newY],[newX, newY],[newX, newY],[newX, newY]])

                inverseHomography = np.linalg.inv(homography)

                car.u_screenPoints = [p for p in carReferencePoints]
                car.screenPoints = [findPointInWarpedImage(p, inverseHomography) for p in carReferencePoints]

                
                totalMovementInPixels = math.sqrt((dx * dx) + (dy * dy))

                if config.HOMOGRAPHY_INFERENCE != "AUTO":
                    metersPerSecond = (totalMovementInPixels * config.FPS) / config.HOMOGRAPHY_SCALING_FACTOR
                    #car.KMH = (metersPerSecond * 60 * 60) / 1000
                    car.recordSpeed((metersPerSecond * 60 * 60) / 1000)
                else:
                    metersPerSecond = (totalMovementInPixels * config.FPS) / 10
                    #car.KMH = (metersPerSecond * 60 * 60) / 1000
                    car.recordSpeed((metersPerSecond * 60 * 60) / 1000)
          
        debugPrint("Tracking complete")

    # Draw the bounding boxes on the screen so that the user can see what's being tracked
    drawingLayer = np.zeros_like(frame)

    if config.MASKRCNN_DRAW_MASKS:
        # Create a full BGR frame containing all of the car masks
        maskRCNNDrawingMask = np.zeros((frameHeight, frameWidth, 3), dtype=np.uint8)
        colorArray = np.array(config.MASKRCNN_DRAW_MASKS_COLOR) / 255

        for car in currentCars:
            # Here's what we have: the bounding box for the car could be slightly offscreen
            # (i.e., it has a start or end coordinate that is negative, or a start or end coordinate
            # that is greater than the frame width/height)
            # We want to be able to fill the entire bounding box with the car's mask (from Mask R-CNN).

            # We first need to convert the mask from grayscale to BGR
            carMask = (cv2.cvtColor(car.mask, cv2.COLOR_GRAY2BGR) * colorArray)
            
            # Make every black pixel transparent 
            # https://stackoverflow.com/questions/70223829/opencv-how-to-convert-all-black-pixels-to-transparent-and-save-it-to-png-file
            # Make a True/False mask of pixels whose BGR values sum to more than zero
            alpha = np.sum(carMask, axis=-1) > 0

            # Convert True/False to 0/255 and change type to "uint8" to match "na"
            alpha = np.uint8(alpha * 255)

            # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
            carMask = np.dstack((carMask, alpha))

            # The car's mask may not be the same size as the bounding box, so we need to resize it.
            # NOTE: If the bounding box has no area, then we can't resize. Abort mission
            if car.boundingBox.getArea() == 0:
                continue
            carMask = cv2.resize(carMask, [car.boundingBox.getWidth(), car.boundingBox.getHeight()])

            # The bounding box may be "spilling" off the edges of the screen in one or more directions.
            # Calculate the spillage, and then we can slice the mask accordingly later.
            startXOffset = max(-car.boundingBox.getX(), 0)
            startYOffset = max(-car.boundingBox.getY(), 0)
            endXOffset = min(frameWidth - car.boundingBox.getEndX(), 0)
            endYOffset = min(frameHeight - car.boundingBox.getEndY(), 0)

            #print(startXOffset, startYOffset, endXOffset, endYOffset)

            # Now, finally, add the mask to the drawing layer.
            #maskRCNNDrawingMask[
            #    car.boundingBox.getY() + startYOffset : car.boundingBox.getEndY() + endYOffset,
            #    car.boundingBox.getX() + startXOffset : car.boundingBox.getEndX() + endXOffset
            #] = carMask[
            #    startYOffset : car.boundingBox.getHeight() + endYOffset,
            #    startXOffset : car.boundingBox.getWidth() + endXOffset
            #]
            # https://stackoverflow.com/a/71701023

            bg_h, bg_w, bg_channels = maskRCNNDrawingMask.shape
            fg_h, fg_w, fg_channels = carMask.shape

            foreground = carMask[
                startYOffset : car.boundingBox.getHeight() + endYOffset,
                startXOffset : car.boundingBox.getWidth() + endXOffset
            ]

            # separate alpha and color channels from the foreground image
            foreground_colors = foreground[:, :, :3]
            alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

            # construct an alpha_mask that matches the image shape
            alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

            background_subsection = maskRCNNDrawingMask[
                car.boundingBox.getY() + startYOffset : car.boundingBox.getEndY() + endYOffset,
                car.boundingBox.getX() + startXOffset : car.boundingBox.getEndX() + endXOffset
            ]

            # combine the background with the overlay image weighted by alpha
            composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

            # overwrite the section of the background image that has been updated
            maskRCNNDrawingMask[
                car.boundingBox.getY() + startYOffset : car.boundingBox.getEndY() + endYOffset,
                car.boundingBox.getX() + startXOffset : car.boundingBox.getEndX() + endXOffset
            ] = composite
            

        # Combine the frame containing the masks to the original frame
        frame = cv2.addWeighted(frame, 1 - config.MASKRCNN_DRAW_MASKS_INTENSITY, maskRCNNDrawingMask.astype("uint8"), config.MASKRCNN_DRAW_MASKS_INTENSITY, 0)

    if config.HOMOGRAPHY_INFERENCE != "AUTO" and config.DRAW_MANUAL_HOMOGRAPHY:
        # Draw the bounding boxes for the cars on the warped image as well
        for car in currentCars:
            start = findPointInWarpedImage((car.boundingBox.getX(), car.boundingBox.getY()), manualHomographyMatrix)
            end = findPointInWarpedImage((car.boundingBox.getEndX(), car.boundingBox.getEndY()), manualHomographyMatrix)
            minX = min(start[0], end[0])
            minY = min(start[1], end[1])
            maxX = max(start[0], end[0])
            maxY = max(start[1], end[1])
            cv2.rectangle(warpedFrame, (minX, minY), (maxX, maxY), car.color, thickness = config.DRAWING_THICKNESS)

            for point in car.u_screenPoints:
                x = point[0]# + car.boundingBox.getX()
                y = point[1]# + car.boundingBox.getY()
                cv2.circle(warpedFrame, (int(x), int(y)), int(manualHomography_pointSize/2), car.color, -1)

        # If the user loaded the homography from a file, then we don't have access
        # to the original points - just the resulting matrix. Ignore them!
        if config.HOMOGRAPHY_INFERENCE == "MANUAL":
            for i in range(len(manualHomography_points)):
                point = manualHomography_points[i]
                color = [(255,100,100),(100,100,255),(100,175,255),(100,255,100)][i]
                warpedPoint = findPointInWarpedImage(point, manualHomographyMatrix)
                cv2.circle(warpedFrame, warpedPoint, manualHomography_pointSize, color, -1)
                cv2.circle(warpedFrame, warpedPoint, int(manualHomography_pointSize/2), [c/3 for c in color], -1)

        cv2.imshow("Vehicle Tracking (Bird's Eye)",warpedFrame)

    # Draw the labels last, so that they're on top of everything else in the image
    for car in currentCars:
        if config.DRAW_3D_BOUNDING_BOXES and len(car.screenPoints) > 0 and car.boundingBox.getArea() > 0:
            # back-left, back-right, front-right, front-left
            backLeft = car.screenPoints[0]
            backRight = car.screenPoints[1]
            frontRight = car.screenPoints[2]
            frontLeft = car.screenPoints[3]

            height = car.height

            for point in car.screenPoints:
                x = point[0]# + car.boundingBox.getX()
                y = point[1]# + car.boundingBox.getY()
                cv2.circle(frame, (x, y - height), int(manualHomography_pointSize/2), car.color, -1)
                cv2.circle(frame, (x, y + height), int(manualHomography_pointSize/2), car.color, -1)
                cv2.line(frame, (x, y - height), (x, y + height), car.color, 2)

            for yOffset in (-height, height):
                for pointA in (backLeft, backRight, frontRight, frontLeft):
                    for pointB in (backLeft, backRight, frontRight, frontLeft):
                        if pointA[0] == pointB[0] and pointA[1] == pointB[1]:
                            continue
                        cv2.line(frame, (pointA[0], pointA[1] + yOffset), (pointB[0], pointB[1] + yOffset), car.color, 2)

        if config.DRAW_2D_BOUNDING_BOXES:
            cv2.rectangle(frame, (car.boundingBox.getX(), car.boundingBox.getY()), (car.boundingBox.getEndX(), car.boundingBox.getEndY()), car.color, thickness = config.DRAWING_THICKNESS)
        
        if config.DRAW_LABELS:
            # This could be done in a more compact way, but in the interest
            # of keeping the code readable:
            text = "ID: {ID} ; KMH: {KMH:.2f}".format(ID=car.ID, KMH=car.getKMH())
            if config.CONVERT_TO_MPH:
                text = "ID: {ID} ; MPH: {MPH:.2f}".format(ID=car.ID, MPH=(car.getKMH() * 0.621371))
                
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
    if config.SAVE_RESULT_AS_VIDEO:
        out.write(frame)

    cv2.imshow("Vehicle Tracking", frame)
    inputKey = cv2.waitKey(int((1 / config.FPS) * 1000)) & 0xFF
    if inputKey == 27 or inputKey == ord("q") or inputKey == ord("Q"): # Esc or Q to quit
        break

    # Save the current gray frame for next iteration
    lastGrayFrame = grayFrame

cap.release()
if config.SAVE_RESULT_AS_VIDEO:
    out.release()
cv2.destroyAllWindows()
