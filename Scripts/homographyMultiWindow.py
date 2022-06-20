# MULTI-WINDOW VERSION
# Script that allows a user to select points on a plane in an image, and will
# use homography to alter the second image.

import cv2
import numpy as np
import math

inputImagePath = "truck.jpg"

image = cv2.imread(inputImagePath)

points = [[50,50],[100,50],[50,100],[100,100]]
selectedPoint = None
pointSize = 10
def onMouse(event, x, y, flags, params):
    global selectedPoint
    selectedDistance = 0

    newImage = image.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        selectedPoint = None
        for point in points:
            distance = math.hypot(x - point[0], y - point[1])
            if math.hypot(x - point[0], y - point[1]) <= pointSize:
                if selectedPoint and selectedDistance <= distance:
                    pass
                else:
                    selectedPoint = point
                    selectedDistance = distance
    elif event == cv2.EVENT_LBUTTONUP:
        selectedPoint = None
        selectedDistance = 0

    if selectedPoint != None:
        selectedPoint[0] = x
        selectedPoint[1] = y

    for i in range(len(points)):
        point = points[i]
        color = [(255,100,100),(100,255,100),(100,100,255),(255,255,100)][i]
        cv2.circle(newImage,(point[0], point[1]),pointSize,color,-1)
        cv2.circle(newImage,(point[0], point[1]),pointSize-4,[c/3 for c in color],-1)

    cv2.imshow("image", newImage)

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    #windowPoints = np.array([[0,0],[imageWidth-1,0],[0,imageHeight-1],[imageWidth-1,imageHeight-1]])
    windowPoints = np.array([[0,0],[50,0],[0,150],[50,150]]) + np.array([[500,500],[500,500],[500,500],[500,500]])

    imagePoints = np.array(points)

    h, status = cv2.findHomography(imagePoints, windowPoints)

    warpedImage = cv2.warpPerspective(image, h, (imageWidth*2,imageHeight*2))
    cv2.imshow("warpedImage", warpedImage)

cv2.imshow("image", image)
cv2.setMouseCallback("image", onMouse)
cv2.waitKey(0)

# Detect the truck in the image to find a bounding box


cv2.destroyAllWindows()