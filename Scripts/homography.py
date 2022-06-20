# Script that allows a user to select points on a plane in an image, and will
# use homography to blow up that region of the image to the full window

import cv2
import numpy as np
import math

inputImagePath = "library2.jpg"

image = cv2.imread(inputImagePath)

points = []
pointSize = 10
def onClick(event, x, y, flags, params):
    newImage = image.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        toDelete = []
        for point in points:
            if math.hypot(x - point[0], y - point[1]) <= pointSize:
                toDelete.append(point)
        for point in toDelete:
            points.remove(point)

    for point in points:
        cv2.circle(newImage,(point[0], point[1]),pointSize,(255,100,100),-1)
        cv2.circle(newImage,(point[0], point[1]),pointSize-4,(255,0,0),-1)

    cv2.imshow("image", newImage)

cv2.imshow("image", image)
cv2.setMouseCallback("image", onClick)
cv2.waitKey(0)

imageWidth = image.shape[1]
imageHeight = image.shape[0]
windowPoints = np.array([[0,0],[imageWidth-1,0],[0,imageHeight-1],[imageWidth-1,imageHeight-1]])

lowerLeft = points[1]
lowerRight = points[2]
upperLeft = points[0]
upperRight = points[3]

lowerLeftAngle = 9999
lowerRightAngle = 9999
upperLeftAngle = 9999
upperRightAngle = 9999

centerX = sum(point[0] for point in points) / 4
centerY = sum(point[1] for point in points) / 4

# Automatically detect the corners of the selection to correctly orient the image
# Upper-left is closest to -135 degrees
# Lower-left is closest to 135 degrees
# Lower-right is closest to 45 degrees
# Upper-right is closest to -45 degrees
for point in points:
    angle = math.degrees(math.atan2(point[1]-centerY, point[0]-centerX))
    if math.fabs(angle - -135) < math.fabs(upperLeftAngle - -135):
        upperLeftAngle = angle
        upperLeft = point
    if math.fabs(angle - 135) < math.fabs(lowerLeftAngle - 135):
        lowerLeftAngle = angle
        lowerLeft = point
    if math.fabs(angle - 45) < math.fabs(lowerRightAngle - 45):
        lowerRightAngle = angle
        lowerRight = point
    if math.fabs(angle - -45) < math.fabs(upperRightAngle - -45):
        upperRightAngle = angle
        upperRight = point

imagePoints = np.array([upperLeft,upperRight,lowerLeft,lowerRight])

h, status = cv2.findHomography(imagePoints, windowPoints)

warpedImage = cv2.warpPerspective(image, h, (imageWidth,imageHeight))
cv2.setMouseCallback("image", lambda event, x, y, flags, params: None)
cv2.imshow("image", warpedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()