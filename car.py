import random
import math

import config

randomColors = [(255,0,0), (0,255,0), (0,0,255)]

class Car():
    def __init__(self,ID):
        self.color = random.choice(randomColors)
        self.ID = ID

        self.boundingBoxHistory = []
        self.maskHistory = []
        self.KMHHistory = []

        # For stitching
        self.firstFrame = None
        self.lastFrame = None

    def __repr__(self):
        return f"Car(ID = {self.ID}, color = {self.color}, boundingBox = {self.boundingBox})"

    def recordBoundingBox(self, boundingBox):
        self.boundingBoxHistory.append(boundingBox)

    def replaceBoundingBox(self, boundingBox):
        self.boundingBoxHistory[-1] = boundingBox

    def getBoundingBox(self):
        return self.boundingBoxHistory[-1]

    def getBoundingBoxHistory(self):
        return self.boundingBoxHistory

    def recordKMH(self, kmh):
        self.KMHHistory.append(kmh)

    def getKMHHistory(self):
        return self.KMHHistory

    def recordMask(self, mask):
        self.maskHistory.append(mask)

    def replaceMask(self, mask):
        self.maskHistory[-1] = mask

    def getMask(self):
        return self.maskHistory[-1]

    def getMaskHistory(self):
        return self.maskHistory