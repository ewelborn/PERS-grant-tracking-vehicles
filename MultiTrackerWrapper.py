import cv2
from copy import deepcopy

import config
from car import Car
from boundingBox import BoundingBox

# Takes in point p in the form of (x, y) and matrix is a 3x3 homography matrix
# https://stackoverflow.com/questions/57399915/how-do-i-determine-the-locations-of-the-points-after-perspective-transform-in-t
def findPointInWarpedImage(p, matrix):
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return (int(px), int(py))

class MultiTrackerWrapper():
    multiTracker = None

    def __init__(self):
        self.multiTracker = cv2.legacy.MultiTracker_create()

    def trackVehiclesInFrames(self, frames, detectedVehicles):
        # Copy the vehicles, don't mutate them
        vehicles = deepcopy(detectedVehicles)

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

        for vehicle in vehicles:
            bb = vehicle.getBoundingBox()
            self.multiTracker.add(trackingModels.get(config.BOUNDING_BOX_TRACKING_MODEL)(), frames[0], [bb.getX(), bb.getY(), bb.getWidth(), bb.getHeight()])

        self.multiTracker.update(frames[0])

        # back-left, back-right, front-right, front-left
        # https://www.thezebra.com/resources/driving/average-car-size/
        # Average car length: 14.7 feet --> 4.48 meters
        # Average car width: 5.8 feet --> 1.77 meters
        MEAN_SEDAN_BOUNDING_BOX = [[0, 0],[1.77 * config.HOMOGRAPHY_SCALING_FACTOR, 0],[1.77 * config.HOMOGRAPHY_SCALING_FACTOR, 4.48 * config.HOMOGRAPHY_SCALING_FACTOR],[0, 4.48 * config.HOMOGRAPHY_SCALING_FACTOR]]

        #print(range(1,len(frames)))
        for i in range(1,len(frames)):
            frame = frames[i]
            #print(frame.shape)
            success, boxes = self.multiTracker.update(frame)

            for i in range(len(vehicles)):
                car = vehicles[i]
                box = boxes[i]
                #print([car.boundingBox.getX(),car.boundingBox.getY(),car.boundingBox.getWidth(),car.boundingBox.getHeight()],"vs",box)
            
                oldBoundingBox = car.getBoundingBox()
                newBoundingBox = BoundingBox(box[0], box[1], box[2], box[3])
                #print(newBoundingBox)
                if newBoundingBox.getArea() != 0 and oldBoundingBox is not None:
                    # Check and make sure the car wasn't lost! When cars go off the edge
                    # of the frame, their bounding boxes have a tendency to go wild. If
                    # the bounding box has a sudden jerk of acceleration in several
                    # directions, then that's a sign that we've probably lost the car.
                    positions = [(bb.getCenterX(), bb.getCenterY()) for bb in car.getBoundingBoxHistory()]
                    velocities = [(p[1][0] - p[0][0], p[1][1] - p[0][1]) for p in zip(positions, positions[1:])]
                    accelerations = [(p[1][0] - p[0][0], p[1][1] - p[0][1]) for p in zip(velocities, velocities[1:])]
                    normalizedAccelerations = [(p[0]**2 + p[1]**2) ** 0.5 for p in accelerations]
                    #print((str(max(normalizedAccelerations)) + "\n") if len(normalizedAccelerations) > 0 else "", end="")

                    if len(normalizedAccelerations) > 0 and max(normalizedAccelerations) > 50:
                        car.recordBoundingBox(BoundingBox(0, 0, 0, 0))
                        continue

                car.recordBoundingBox(newBoundingBox)

        return vehicles