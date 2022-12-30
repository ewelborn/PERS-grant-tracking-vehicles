import cv2
import numpy as np

import config
from car import Car
from boundingBox import BoundingBox

class MaskRCNNDetector():
    model = None

    def __init__(self):
        self.model = cv2.dnn.readNetFromTensorflow(config.MASKRCNN_WEIGHTS_PATH, config.MASKRCNN_CONFIG_PATH)

    '''
    todo
    '''
    def findVehiclesInFrame(self, frame, currentCarID=0):
        COCOLabels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
        imageBlob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

        # Propagate and detect objects in the image
        self.model.setInput(imageBlob)
        boundingBoxes, masks = self.model.forward(["detection_out_final", "detection_masks"])

        # Use non-maximum suppression to avoid generate multiple bounding boxes
        # for the same object
        boxesList = [boundingBoxes[0, 0, i, 3:7] for i in range(0, boundingBoxes.shape[2])]
        confidencesList = [boundingBoxes[0, 0, i, 2] for i in range(0, boundingBoxes.shape[2])]

        # NMSBoxes takes two special arguments - score_threshold and nms_threshold
        # score_threshold is used to filter out low confidence results (i.e. it's the minimum
        #   confidence necessary to keep a result)
        # nms_threshold is the maximum intersection allowed between two results
        maxValueIDs = cv2.dnn.NMSBoxes(boxesList, confidencesList, config.DETECTION_MINIMUM_CONFIDENCE, config.NMS_THRESHOLD)

        currentCars = []
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        for i in maxValueIDs:
            predictedClassID = int(boundingBoxes[0, 0, i, 1])
            predictedClassLabel = COCOLabels[predictedClassID] if predictedClassID < len(COCOLabels) else ""
            predictionConfidence = boundingBoxes[0, 0, i, 2]

            if predictedClassLabel in config.COCO_LABELS_TO_DETECT:
                car = Car(currentCarID)
                x, y, endX, endY = (boundingBoxes[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])).astype("int")
                car.recordBoundingBox(BoundingBox(x, y, endX - x, endY - y))

                currentCarID += 1

                currentCars.append(car)

                # Find the pixel mask of the car
                mask = masks[i, predictedClassID]
                mask = cv2.resize(mask, (car.getBoundingBox().getWidth(), car.getBoundingBox().getHeight()), interpolation=cv2.INTER_CUBIC)
                mask = (mask > config.MASKRCNN_PIXEL_SEGMENTATION_THRESHOLD).astype("uint8") * 255
                car.mask = mask

        return currentCars