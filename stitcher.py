from copy import deepcopy

import config

def stitchTrackedVehicles(trackedVehiclesList):
    cumulativeTrackedVehicles = deepcopy(trackedVehiclesList[0])

    currentVehicleID = len(cumulativeTrackedVehicles)
    totalFrames = 0

    for vehicle in cumulativeTrackedVehicles:
        vehicle.firstFrame = 0

    for i in range(1, len(trackedVehiclesList)):
        # Every batch of tracked vehicles advances the video by refresh_rate - 1 frames
        totalFrames += config.DETECTION_REFRESH_RATE - 1

        # Assume that all previous vehicles will be lost - then try to prove the opposite
        for vehicle in cumulativeTrackedVehicles:
            if vehicle.lastFrame is None:
                vehicle.lastFrame = totalFrames

        # Try to match vehicles that we've seen so far with newly detected vehicles
        for vehicle in trackedVehiclesList[i]:
            bestMatch = None
            bestOverlap = 0

            # Only look at vehicles that are still being tracked successfully (not old, lost vehicles)
            for otherVehicle in [vehicle for vehicle in cumulativeTrackedVehicles if vehicle.lastFrame == totalFrames]:
                # Check the last frame of the cumulative vehicle and the first frame of the new vehicle.
                #   (These frames are overlapping)
                overlap = otherVehicle.getBoundingBox().getOverlap(vehicle.getBoundingBoxHistory()[0])

                if overlap >= config.MINIMUM_BB_OVERLAP and ((bestMatch is None) or (overlap > bestOverlap)):
                    bestMatch = otherVehicle
                    bestOverlap = overlap

            if bestMatch is None:
                # This is probably a new vehicle, give it a unique ID and add it to the list
                vehicle.ID = currentVehicleID
                currentVehicleID += 1

                vehicle.firstFrame = totalFrames

                cumulativeTrackedVehicles.append(vehicle)
            else:
                # This is probably a vehicle that we've detected before. Add its information to the
                # previous vehicle's information

                # Replace the bounding box in the overlapping frame with the one from our new 
                #   batch - it's likely more accurate.
                bbHistory = vehicle.getBoundingBoxHistory()
                bestMatch.replaceBoundingBox(bbHistory[0])
                for i in range(1, len(bbHistory)):
                    bestMatch.recordBoundingBox(bbHistory[i])

                if len(bestMatch.getBoundingBoxHistory()) == 14:
                    print(len(bbHistory))

                # Flag the vehicle so that the algorithm is aware it's still actively tracked
                bestMatch.lastFrame = None

                pass
    
    # Any vehicles that are left should have the last frame properly set by the caller - we don't
    #   know which frame the video ended on.

    return cumulativeTrackedVehicles