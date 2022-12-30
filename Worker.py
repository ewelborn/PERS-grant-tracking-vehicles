import cv2
import numpy as np
import copy
import math

import config

from car import Car
from boundingBox import BoundingBox

# This class represents an independent detection/tracking system that
# can be easily parallelized.
class Worker:
    vehicleDetector = None
    vehicleTracker = None

    def __init__(self, vehicleDetector, vehicleTracker):
        self.vehicleDetector = vehicleDetector
        self.vehicleTracker = vehicleTracker

    # This function's job is to take in a sequence of 1 or more frames,
    # detect vehicles in the first frame, and track them over the remaining
    # frames. Then, return a list of vehicles and their positions for
    # each frame.
    def findVehiclesInFrames(self, frames):
        ### DETECT
        detectedVehicles = self.vehicleDetector.findVehiclesInFrame(frames[0])

        ### TRACK
        trackedVehicles = self.vehicleTracker.trackVehiclesInFrames(frames, detectedVehicles)

        return trackedVehicles