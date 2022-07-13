# Script for importing an image sequence and exporting it as a video file
import cv2

imageSequencePath = "thermal_13Aug2020\\"
imageNameTemplate = "frame_1_{i}.jpg"
startIndex = 2500 # Inclusive
endIndex = 3528 # Inclusive

outputVideoPath = "thermalSequence_trimmed.mp4"
outputFPS = 54

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

for i in range(startIndex, endIndex + 1):
    if i % 100 == 0:
        print(i - startIndex,"/",endIndex - startIndex + 1,"frames left")

    frame = cv2.imread(imageSequencePath + imageNameTemplate.format(i=i))

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    if out is None:
        out = cv2.VideoWriter(outputVideoPath, fourcc, outputFPS, (frameWidth, frameHeight))

    out.write(frame)

print("finished encoding")

out.release()
