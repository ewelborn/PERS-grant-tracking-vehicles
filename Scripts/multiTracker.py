# This is a test script to try OpenCV's legacy multitracker system
# Tested on opencv-contrib-python 4.6.0.66

import cv2

videoPath = "Datasets\\cam_10.mp4"

cap = cv2.VideoCapture(videoPath)
ret, frame = cap.read()

multiTracker = cv2.legacy.MultiTracker_create()

while True:
    ROI = cv2.selectROI("Frame",frame)
    if sum([a for a in ROI]) == 0:
        break
    multiTracker.add(cv2.legacy.TrackerKCF_create(),frame,ROI)
    print(ROI)

while True:
    ret, frame = cap.read()

    success, boxes = multiTracker.update(frame)
    for i, newBox in enumerate(boxes):
        print(newBox)
        p1 = (int(newBox[0]),int(newBox[1]))
        p2 = (int(newBox[0])+int(newBox[2]),int(newBox[1])+int(newBox[3]))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(int((1/10)*1000)) == ord("c"):
        break