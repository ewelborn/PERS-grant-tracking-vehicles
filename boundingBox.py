class BoundingBox():
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def __str__(self):
        return "[{x}, {y}, {width}, {height}]".format(x=self.x, y=self.y, width=self.width, height=self.height)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getCenterX(self):
        return int(self.x + (self.width / 2))

    def getCenterY(self):
        return int(self.y + (self.height / 2))

    def getEndX(self):
        return self.x + self.width

    def getEndY(self):
        return self.y + self.height

    def getArea(self):
        return self.width * self.height

    def getOverlap(bb1,bb2):
        # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
        x = max(bb1.getX(), bb2.getX())
        y = max(bb1.getY(), bb2.getY())
        endX = min(bb1.getEndX(), bb2.getEndX())
        endY = min(bb1.getEndY(), bb2.getEndY())

        if endX < x or endY < y:
            return 0.0

        overlapArea = (endX - x) * (endY - y)

        return overlapArea / float(bb1.getArea() + bb2.getArea() - overlapArea)