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
if not(config.DETECTION_MODEL in ["YOLO","MASKRCNN"]):
    print(config.DETECTION_MODEL,"is not a valid detection model")
    print("Please try \"YOLO\" or \"MASKRCNN\"")
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

def debugPrint(*args):
    if config.DEBUG:
        if config.DEBUG_TIMECODE:
            global totalFrames
            timeSignature = "({seconds}:{frames})".format(seconds=totalFrames//config.FPS, frames=totalFrames%config.FPS)
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

multiTracker = cv2.legacy.MultiTracker_create()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# Keep reading frames from the video until it's over
progress = 0
firstFrame = True
lastGrayFrame = None
totalFrames = 0
while True:
    progress += 1
    ret, frame = cap.read()

    totalFrames += 1

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    if config.SAVE_RESULT_AS_VIDEO and (out is None):
        out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, config.FPS, (frameWidth, frameHeight))

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if config.MANUAL_HOMOGRAPHY and firstFrame:
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
        progress = config.DETECTION_REFRESH_RATE

    firstFrame = False

    if ret == False:
        print("Video EOF")
        break

    if progress >= config.DETECTION_REFRESH_RATE:
        debugPrint(config.DETECTION_MODEL, "detecting...")

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

                currentCars.append(car)

                # Find points on the car that can be used for optical flow tracking
                mask = np.zeros_like(grayFrame)
                # KEEP IN MIND!! The frame is in the form of [Y][X]
                mask[car.boundingBox.getY():car.boundingBox.getEndY(), car.boundingBox.getX():car.boundingBox.getEndX()] = 255
                car.opticalFlowPoints = cv2.goodFeaturesToTrack(grayFrame, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7)
        
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
                predictedClassLabel = COCOLabels[predictedClassID]
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

                    currentCars.append(car)

                    # Find the pixel mask of the car
                    mask = masks[i, predictedClassID]
                    mask = cv2.resize(mask, (car.boundingBox.getWidth(), car.boundingBox.getHeight()), interpolation=cv2.INTER_CUBIC)
                    mask = (mask > config.MASKRCNN_PIXEL_SEGMENTATION_THRESHOLD).astype("uint8") * 255
                    car.mask = mask

                    # Register the car's bounding box with the multitracker
                    # Use the tracking model that the user specified in the config settings
                    multiTracker.add(trackingModels.get(config.BOUNDING_BOX_TRACKING_MODEL)(), frame, [x, y, endX - x, endY - y])

        debugPrint(config.DETECTION_MODEL, "detection is complete")
        if config.MANUAL_HOMOGRAPHY == False:
            # Create homographies for the cars
            debugPrint("Calculating homographies...")
        
            for car in currentCars:
                # VERY PRIMITIVE: We'll use the four corners of the bounding boxes
                # to try and align the car to a mean 3:1 size (length:width)
                # https://www.thezebra.com/resources/driving/average-car-size/
                # Average car length: 14.7 feet --> 4.48 meters
                # Average car width: 5.8 feet --> 1.77 meters
                carReferencePoints = np.array([[0, 0],[17.7, 0],[17.7, 44.8],[0, 44.8]])
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
    warpedFrame = cv2.warpPerspective(frame, manualHomographyMatrix, (frameWidth, frameHeight)) if config.MANUAL_HOMOGRAPHY else 1

    debugPrint("Tracking with {model}...".format(model=config.BOUNDING_BOX_TRACKING_MODEL))
    if not (lastGrayFrame is None):
        success, boxes = multiTracker.update(frame)

        for i in range(len(currentCars)):
            car = currentCars[i]
            box = boxes[i]
            #print([car.boundingBox.getX(),car.boundingBox.getY(),car.boundingBox.getWidth(),car.boundingBox.getHeight()],"vs",box)
            
            oldBoundingBox = car.boundingBox
            car.boundingBox = BoundingBox(box[0], box[1], box[2], box[3])
            if car.boundingBox.getArea() != 0 and oldBoundingBox is not None:
                homography = manualHomographyMatrix if config.MANUAL_HOMOGRAPHY else car.homographyMatrix

                # Estimate the car's speed based on the movement of
                # the bounding box in the top-down image
                oldX, oldY = findPointInWarpedImage((oldBoundingBox.getCenterX(), oldBoundingBox.getCenterY()), homography)
                newX, newY = findPointInWarpedImage((car.boundingBox.getCenterX(), car.boundingBox.getCenterY()), homography)
                dx = newX - oldX
                dy = newY - oldY

                totalMovementInPixels = math.sqrt((dx * dx) + (dy * dy))

                if config.MANUAL_HOMOGRAPHY:
                    metersPerSecond = (totalMovementInPixels * config.FPS) / manualHomography_pixelsToMeters
                    car.KMH = (metersPerSecond * 60 * 60) / 1000
                else:
                    metersPerSecond = (totalMovementInPixels * config.FPS) / 10
                    car.KMH = (metersPerSecond * 60 * 60) / 1000

                    
    debugPrint("Tracking complete")

    # Draw the bounding boxes on the screen so that the user can see what's being tracked
    drawingLayer = np.zeros_like(frame)

    if config.MASKRCNN_DRAW_MASKS:
        # Create a full frame containing all of the car masks
        maskRCNNDrawingMask = np.zeros_like(frame)
        colorArray = np.array(config.MASKRCNN_DRAW_MASKS_COLOR) / 255

        for car in currentCars:
            # Naive fix for out of bounds errors: just don't draw the mask if it would go out of bounds
            if (car.boundingBox.getY() + car.mask.shape[0] >= frameHeight) or (car.boundingBox.getX() + car.mask.shape[1] >= frameWidth):
                continue
            # Also, don't draw the mask if we lost the car (if we don't do this, a phantom mask gets
            # drawn in the upper-left corner of the screen)
            elif car.boundingBox.getArea() == 0:
                continue
            maskRCNNDrawingMask[car.boundingBox.getY():car.boundingBox.getY()+car.mask.shape[0], car.boundingBox.getX():car.boundingBox.getX()+car.mask.shape[1]] = (cv2.cvtColor(car.mask, cv2.COLOR_GRAY2BGR) * colorArray)
        
        # Combine the frame containing the masks to the original frame
        frame = cv2.addWeighted(frame, 1 - config.MASKRCNN_DRAW_MASKS_INTENSITY, maskRCNNDrawingMask.astype("uint8"), config.MASKRCNN_DRAW_MASKS_INTENSITY, 0)

    if config.MANUAL_HOMOGRAPHY and config.DRAW_MANUAL_HOMOGRAPHY:
        # Draw the bounding boxes for the cars on the warped image as well
        for car in currentCars:
            start = findPointInWarpedImage((car.boundingBox.getX(), car.boundingBox.getY()), manualHomographyMatrix)
            end = findPointInWarpedImage((car.boundingBox.getEndX(), car.boundingBox.getEndY()), manualHomographyMatrix)
            minX = min(start[0], end[0])
            minY = min(start[1], end[1])
            maxX = max(start[0], end[0])
            maxY = max(start[1], end[1])
            cv2.rectangle(warpedFrame, (minX, minY), (maxX, maxY), car.color, thickness = config.DRAWING_THICKNESS)

        for i in range(len(manualHomography_points)):
            point = manualHomography_points[i]
            color = [(255,100,100),(100,100,255),(100,255,100),(100,175,255)][i]
            warpedPoint = findPointInWarpedImage(point, manualHomographyMatrix)
            cv2.circle(warpedFrame,warpedPoint,manualHomography_pointSize,color,-1)
            cv2.circle(warpedFrame,warpedPoint,int(manualHomography_pointSize/2),[c/3 for c in color],-1)

        cv2.imshow("Vehicle Tracking (Bird's Eye)",warpedFrame)

    # Draw the labels last, so that they're on top of everything else in the image
    for car in currentCars:
        if config.DRAW_BOUNDING_BOXES:
            cv2.rectangle(drawingLayer, (car.boundingBox.getX(), car.boundingBox.getY()), (car.boundingBox.getEndX(), car.boundingBox.getEndY()), car.color, thickness = config.DRAWING_THICKNESS)
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
