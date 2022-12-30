class ManualHomographyWrapper():
    homography = None

    def __init__(self, homography):
        self.homography = homography

    def getHomographyFromFrame(self, frame):
        # Don't estimate anything, just return our manual estimation
        return self.homography