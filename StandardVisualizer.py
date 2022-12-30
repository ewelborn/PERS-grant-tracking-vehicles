import numpy as np
import cv2

import config

class StandardVisualizer():
    def drawFrames(self, frames, trackedVehicles):
        newFrames = []

        framesSinceLabelUpdate = 0
        for i in range(len(frames)):
            newFrame = np.array(frames[i])

            for vehicle in trackedVehicles:
                if not(vehicle.firstFrame <= i <= vehicle.lastFrame):
                    # Vehicle does not exist in this part of the video
                    continue

                j = i - vehicle.firstFrame
                bbHistory = vehicle.getBoundingBoxHistory()
                bb = bbHistory[j]
                kmh = vehicle.getKMHHistory()[j]

                #print(f"j {j}, bbHistory {len(bbHistory)}, firstFrame {vehicle.firstFrame}, lastFrame {vehicle.lastFrame}")

                if config.DRAW_2D_BOUNDING_BOXES:    
                    cv2.rectangle(
                        newFrame, 
                        (bb.getX(), bb.getY()), (bb.getEndX(), bb.getEndY()), 
                        vehicle.color, 
                        thickness = config.DRAWING_THICKNESS
                    )

                if config.DRAW_LABELS:
                    # This could be done in a more compact way, but in the interest
                    # of keeping the code readable:
                    text = "ID: {ID} ; KMH: {KMH:.2f}".format(ID=vehicle.ID, KMH=kmh)
                    if config.CONVERT_TO_MPH:
                        text = "ID: {ID} ; MPH: {MPH:.2f}".format(ID=vehicle.ID, MPH=(kmh * 0.621371))
                
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    thickness = 2
                    textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
                    #cv2.rectangle(drawingLayer, (car.boundingBox.getX() - textSize[0], car.boundingBox.getY() - textSize[1]), (car.boundingBox.getX(), car.boundingBox.getY()), (255,0,0), thickness = DRAWING_THICKNESS)
                    newFrame[bb.getY() - textSize[1]:bb.getY(), bb.getX():bb.getX() + textSize[0]] = (0,0,0)
                    cv2.putText(newFrame, text, (bb.getX(), bb.getY()), fontFace, fontScale, (255,255,255), thickness)

            newFrames.append(newFrame)

        return newFrames