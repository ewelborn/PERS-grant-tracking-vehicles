# Primitive script for tracking cars in a video with YOLOv4/COCO and KLT optical flow

import numpy as np
import cv2
import math

cap = cv2.VideoCapture("\\Datasets\\slow_traffic_small.mp4")
ret, old_frame = cap.read()

### Use the first frame of the video to detect cars for tracking
image_height = old_frame.shape[0]
image_width = old_frame.shape[1]

class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors,(16,1)) #np.tile(class_colors,(16,1))

# Parameters for ShiTomasi corner detection
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance = 7,
    blockSize = 7)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

nms_boxes_list = []
cars = []
carID = 0
old_gray = None
def detectCars(old_frame):
    image_blob = cv2.dnn.blobFromImage(old_frame,0.003922,(416,416),swapRB=True,crop=False)

    class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

    yolo_model = cv2.dnn.readNetFromDarknet("yolov4.cfg","yolov4.weights")
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer-1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    yolo_model.setInput(image_blob)
    object_detection_layers = yolo_model.forward(yolo_output_layer)

    class_ids_list = []
    boxes_list = []
    confidences_list = []

    for object_detection_layer in object_detection_layers:
        for object_detection in object_detection_layer:
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]

            if prediction_confidence > 0.2:
                predicted_class_label = class_labels[predicted_class_id]

                bounding_box = object_detection[0:4] * np.array([image_width,image_height,image_width,image_height])
                (box_center_x_point, box_center_y_point, box_width, box_height) = bounding_box.astype("int")
                start_x_point = int(box_center_x_point - (box_width / 2))
                start_y_point = int(box_center_y_point - (box_height / 2))

                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_point,start_y_point,int(box_width),int(box_height)])

    max_value_ids = cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)
    
    nms_boxes_list.clear()
    for max_value_id in max_value_ids:
        max_class_id = max_value_id
        box = boxes_list[max_class_id]
        start_x_point = box[0]
        start_y_point = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]

        end_x_point = start_x_point + box_width
        end_y_point = start_y_point + box_height

        box_color = class_colors[predicted_class_id]
        box_color = [int(c) for c in box_color]

        # Only append if it's a car
        if predicted_class_label == "car":
            nms_boxes_list.append([start_x_point,start_y_point,end_x_point,end_y_point])

        #predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        #print("predicted object {}".format(predicted_class_label))

    # Take frame and find corners in it (with respect to the detected cars)
    oldCars = cars.copy()
    cars.clear()
    old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    for box in nms_boxes_list:
        car = {}

        car["mask"] = np.zeros_like(old_gray)
        # KEEP IN MIND!! The frame is in the form of [Y][X]
        car["mask"][box[1]:box[3],box[0]:box[2]] = 255
        car["p0"] = cv2.goodFeaturesToTrack(old_gray,mask=car["mask"],**feature_params)

        car["width"] = (box[2] - box[0])
        car["height"] = (box[3] - box[1])
        car["centerX"] = (box[0]) + (car["width"]/2)
        car["centerY"] = (box[1]) + (car["height"]/2)
        global carID
        
        # Is this an old car or a new car?
        found = False
        foundDistance = 0
        for otherCar in oldCars:
            dx = car["centerX"] - otherCar["centerX"]
            dy = car["centerY"] - otherCar["centerY"]
            distance = math.hypot(dx,dy)
            if distance < 100:
                if (found and foundDistance > distance) or (not found):
                    found = otherCar
                    foundDistance = distance

        if found:
            car["ID"] = found["ID"]
        else:
            car["ID"] = carID
            carID += 1

        cars.append(car)

    return cars,old_gray

cars,old_gray = detectCars(old_frame)
### Track the detected cars until the video ends

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Create a mask image for drawing purposes
drawing_mask = np.zeros_like(old_frame)

# How many frames should we wait until we let YOLO detect again?
framesBetweenDetections = 60

currentFramesBetweenDetections = 0

while(True):
    ret,frame = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    currentFramesBetweenDetections += 1
    if currentFramesBetweenDetections >= framesBetweenDetections:
        currentFramesBetweenDetections = 0
        detectCars(frame)
        drawing_mask = np.zeros_like(old_frame)

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Calculate optical flow
    for car in cars:
        p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,car["p0"],None,**lk_params)

        # Select good points
        if p1 is not None:
            car["good_new"] = p1[st==1]
            good_old = car["p0"][st==1]

        # Draw the tracks
        dx = 0
        dy = 0
        for i,(new,old) in enumerate(zip(car["good_new"],good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            drawing_mask = cv2.line(drawing_mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            dx += a - c
            dy += b - d
        img = cv2.add(frame,drawing_mask)

        if len(car["good_new"]) > 0:
            dx = dx / len(car["good_new"])
            dy = dy / len(car["good_new"])
            car["centerX"] += dx
            car["centerY"] += dy
    
            cv2.rectangle(
                frame,
                (int(car["centerX"]-(car["width"]/2)),int(car["centerY"]-(car["height"]/2))),
                (int(car["centerX"]+(car["width"]/2)),int(car["centerY"]+(car["height"]/2))),
                [int(c) for c in class_colors[car["ID"] % len(class_colors)]]
            )

        # Uncomment to see the masks
        #img = cv2.bitwise_and(img,cv2.merge((mask,mask,mask)))

    cv2.imshow("frame",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    for car in cars:
        car["p0"] = car["good_new"].reshape(-1,1,2)

cv2.destroyAllWindows()