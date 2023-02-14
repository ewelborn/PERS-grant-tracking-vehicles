import numpy as np
import math
import cv2

import config

class StandardVisualizer():
    def drawFrames(self, frames, trackedVehicles):
        newFrames = []

        framesSinceLabelUpdate = 0
        for i in range(len(frames)):
            newFrame = np.array(frames[i])

            frameWidth = newFrame.shape[1]
            frameHeight = newFrame.shape[0]

            for vehicle in trackedVehicles:
                if not(vehicle.firstFrame <= i <= vehicle.lastFrame):
                    # Vehicle does not exist in this part of the video
                    continue

                j = i - vehicle.firstFrame
                bbHistory = vehicle.getBoundingBoxHistory()
                bb = bbHistory[j]
                maskHistory = vehicle.getMaskHistory()
                vehicleMask = maskHistory[j]
                kmh = vehicle.getKMHHistory()[j]

                if len(bbHistory) != len(maskHistory):
                    print(len(bbHistory),"vs",len(maskHistory))

                #print(f"j {j}, bbHistory {len(bbHistory)}, firstFrame {vehicle.firstFrame}, lastFrame {vehicle.lastFrame}")

                if config.MASKRCNN_DRAW_MASKS and bb.getArea() > 0:
                    colorArray = np.array(config.MASKRCNN_DRAW_MASKS_COLOR) / 255

                    # Here's what we have: the bounding box for the car could be slightly offscreen
                    # (i.e., it has a start or end coordinate that is negative, or a start or end coordinate
                    # that is greater than the frame width/height)
                    # We want to be able to fill the entire bounding box with the car's mask (from Mask R-CNN).

                    # We first need to convert the mask from grayscale to BGR
                    scaledMask = (cv2.cvtColor(vehicleMask.getScaledMask(), cv2.COLOR_GRAY2BGR) * colorArray)
            
                    # Make every black pixel transparent 
                    # https://stackoverflow.com/questions/70223829/opencv-how-to-convert-all-black-pixels-to-transparent-and-save-it-to-png-file
                    # Make a True/False mask of pixels whose BGR values sum to more than zero
                    alpha = np.sum(scaledMask, axis=-1) > 0

                    # Convert True/False to 0/255 and change type to "uint8" to match "na"
                    alpha = np.uint8(alpha * 255 * config.MASKRCNN_DRAW_MASKS_INTENSITY)

                    # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
                    scaledMask = np.dstack((scaledMask, alpha))


                    # The bounding box may be "spilling" off the edges of the screen in one or more directions.
                    # Calculate the spillage, and then we can slice the mask accordingly later.
                    startXOffset = max(-bb.getX(), 0)
                    startYOffset = max(-bb.getY(), 0)
                    endXOffset = min(frameWidth - bb.getEndX(), 0)
                    endYOffset = min(frameHeight - bb.getEndY(), 0)

                    # Now, finally, add the mask to the drawing layer.
                    #maskRCNNDrawingMask[
                    #    car.boundingBox.getY() + startYOffset : car.boundingBox.getEndY() + endYOffset,
                    #    car.boundingBox.getX() + startXOffset : car.boundingBox.getEndX() + endXOffset
                    #] = carMask[
                    #    startYOffset : car.boundingBox.getHeight() + endYOffset,
                    #    startXOffset : car.boundingBox.getWidth() + endXOffset
                    #]
                    # https://stackoverflow.com/a/71701023

                    foreground = scaledMask[
                        startYOffset : bb.getHeight() + endYOffset,
                        startXOffset : bb.getWidth() + endXOffset
                    ]

                    # separate alpha and color channels from the foreground image
                    foreground_colors = foreground[:, :, :3]
                    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

                    # construct an alpha_mask that matches the image shape
                    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

                    background_subsection = newFrame[
                        bb.getY() + startYOffset : bb.getEndY() + endYOffset,
                        bb.getX() + startXOffset : bb.getEndX() + endXOffset
                    ]

                    # combine the background with the overlay image weighted by alpha
                    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

                    # overwrite the section of the background image that has been updated
                    newFrame[
                        bb.getY() + startYOffset : bb.getEndY() + endYOffset,
                        bb.getX() + startXOffset : bb.getEndX() + endXOffset
                    ] = composite

                    # Combine the frame containing the masks to the original frame
                    #frame = cv2.addWeighted(frame, 1 - config.MASKRCNN_DRAW_MASKS_INTENSITY, maskRCNNDrawingMask.astype("uint8"), config.MASKRCNN_DRAW_MASKS_INTENSITY, 0)

                if config.DRAW_2D_BOUNDING_BOXES:    
                    cv2.rectangle(
                        newFrame, 
                        (bb.getX(), bb.getY()), (bb.getEndX(), bb.getEndY()), 
                        vehicle.color, 
                        thickness = config.DRAWING_THICKNESS
                    )

                    if config.DRAW_ANGLE_OF_MOTION and j >= 6:
                        dx = bb.getCenterX() - bbHistory[j - 6].getCenterX()
                        dy = bb.getCenterY() - bbHistory[j - 6].getCenterY()

                        angleOfMotion = math.atan2(dy, dx)
                        mag = 50

                        # Motion vector
                        pt1 = (int(bb.getCenterX() - math.cos(angleOfMotion) * mag), int(bb.getCenterY() - math.sin(angleOfMotion) * mag))
                        pt2 = (int(bb.getCenterX() + math.cos(angleOfMotion) * mag), int(bb.getCenterY() + math.sin(angleOfMotion) * mag))
                        
                        cv2.arrowedLine(
                            newFrame,
                            pt1,
                            pt2,
                            color=(255,0,0),
                            thickness=5
                        )

                        # Perpendicular vector
                        # We have two perpendicular vectors that are possible - one that points
                        #   upwards and another that points downwards. We need to pick the one
                        #   that points downwards.
                        # That would be on the bottom half of the unit circle - between pi and 2pi
                        proposedAngle = angleOfMotion - (math.pi / 2)
                        if proposedAngle < 0:
                            proposedAngle += math.pi * 2

                        # I have no idea why this works.
                        if math.pi <= proposedAngle <= math.pi * 2:
                            angleOfMotion += (math.pi / 2)
                        else:
                            angleOfMotion -= (math.pi / 2) 
                            
                        #angleOfMotion = angleOfMotion + ((math.pi / 2) * -1 if bb.getCenterX() < (newFrame.shape[1] / 2) else 1)
                        pt1 = (int(bb.getCenterX() - math.cos(angleOfMotion) * mag), int(bb.getCenterY() - math.sin(angleOfMotion) * mag))
                        pt2 = (int(bb.getCenterX() + math.cos(angleOfMotion) * mag), int(bb.getCenterY() + math.sin(angleOfMotion) * mag))

                        cv2.arrowedLine(
                            newFrame,
                            pt1,
                            pt2,
                            color=(0,0,255),
                            thickness=5
                        )

                        # Sort all non-zero pixels in the mask by their distance along the perpendicular axis
                        if not (vehicleMask.getScaledMask() is None):
                            indexArray = np.indices(vehicleMask.getScaledMask().shape)

                            # indexArray's shape is currently (2, width, height), which does not work for
                            #   the calculations we need to do. We're transposing it to (width, height, 2)
                            indexArray = np.transpose(indexArray, (1, 2, 0))

                            v = np.array([math.sin(angleOfMotion), math.cos(angleOfMotion)])

                            distanceArray = indexArray @ v

                            mn = np.min(distanceArray)
                            r = np.max(distanceArray) - mn

                            distanceArray = np.where(vehicleMask.getScaledMask(), distanceArray, -np.inf)
                        
                            # Get only the points that are within 5% of the distance of the farthest point (if that makes sense)
                            farthestPoints = np.argwhere(distanceArray >= np.max(distanceArray) - (r * 0.05))
                            #farthestPoints = np.argsort(distanceArray, axis=None)
                            #print("fp", farthestPoints.shape, farthestPoints)

                            # Add the distance as the third column
                            distanceVector = [distanceArray[p[0]][p[1]] for p in farthestPoints]
                            #print("dv", distanceVector)
                            farthestPoints = np.append(farthestPoints, np.array(distanceVector).reshape((len(distanceVector), 1)), axis=1)
                            #print("fp", farthestPoints)
                            # Sort by distance
                            ind = np.argsort(farthestPoints[:,-1])
                            farthestPoints = farthestPoints[ind]
                            #print("fp", farthestPoints)

                            farthestPoint = farthestPoints[0]

                            cv2.circle(newFrame, [int(farthestPoint[1]) + bb.getX(), int(farthestPoint[0]) + bb.getY()], 5, (255,255,100), 5)

                            # Try and find the other tire. It should be the farthest point that is *not* within a certain
                            #   range of the first tire
                            for point in farthestPoints:
                                if ((point[0] - farthestPoint[0]) ** 2 + (point[1] - farthestPoint[1]) ** 2) ** 0.5 > 15:
                                    cv2.circle(newFrame, [int(point[1]) + bb.getX(), int(point[0]) + bb.getY()], 5, (255,255,255), 5)
                                    break

                            for y in range(distanceArray.shape[0]):
                                for x in range(distanceArray.shape[1]):
                                    continue
                                    if (y + bb.getY() < 0) or (y + bb.getY() >= newFrame.shape[0]) or (x + bb.getX() < 0) or (x + bb.getX() >= newFrame.shape[1]):
                                        continue
                                    if r == 0:
                                        continue
                                    
                                    newFrame[y + bb.getY(), x + bb.getX()] = np.array([255 * ((distanceArray[y][x] - mn) / r),255 * ((distanceArray[y][x] - mn) / r),0])

                                    #print("mn",mn,"r",r,"value",np.array([255 * ((distanceArray[y][x] - mn) / r),255 * ((distanceArray[y][x] - mn) / r),0]),"actual",newFrame[y + bb.getY(), x + bb.getX()])

                if config.DRAW_LABELS:
                    # Check if the vehicle is parked
                    startFrame = max(0, j - config.PARKED_VEHICLE_TIME_THRESHOLD * config.FPS)
                    endFrame = min(j + config.PARKED_VEHICLE_TIME_THRESHOLD * config.FPS, len(vehicle.getKMHHistory()) - 1)
                    window = vehicle.getKMHHistory()[startFrame:endFrame]

                    startBB = vehicle.getBoundingBoxHistory()[startFrame]
                    endBB = vehicle.getBoundingBoxHistory()[min(j + config.PARKED_VEHICLE_TIME_THRESHOLD * config.FPS, len(vehicle.getBoundingBoxHistory()) - 1)]

                    startPoint = np.array([startBB.getCenterX(), startBB.getEndY()])
                    endPoint = np.array([endBB.getCenterX(), endBB.getEndY()])

                    if max(window) < config.PARKED_VEHICLE_SPEED_THRESHOLD:
                        # Vehicle is parked
                        text = "ID: {ID} ; *PARKED*".format(ID=vehicle.ID)
                    elif np.linalg.norm(endPoint - startPoint) < 25:
                        # Vehicle is probably in the background.. can't really get
                        #   a good speed estimation from it
                        text = "ID: {ID} ; *UNKNOWN*".format(ID=vehicle.ID)
                    else:
                        # Vehicle is moving
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

                if config.DRAW_FRAME_COUNT:
                    # Draw the current frame onto the video
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    thickness = 2

                    cv2.putText(newFrame, str(i), (100, 100), fontFace, fontScale, (0,0,0), thickness)

            newFrames.append(newFrame)

        return newFrames