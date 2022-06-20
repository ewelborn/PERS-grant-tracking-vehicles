# Basic YOLOv4/COCO script for detecting cars in a video

import numpy as np
import cv2

file_video_stream  = cv2.VideoCapture("\\Datasets\\slow_traffic_small.mp4")
while file_video_stream.isOpened:
    ret,current_frame = file_video_stream.read()
    image_to_detect = current_frame

    image_height = image_to_detect.shape[0]
    image_width = image_to_detect.shape[1]

    image_blob = cv2.dnn.blobFromImage(image_to_detect,0.003922,(416,416),swapRB=True,crop=False)

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

    class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors,(16,1)) #np.tile(class_colors,(16,1))

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

        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))

        cv2.rectangle(image_to_detect,(start_x_point,start_y_point),(end_x_point,end_y_point),box_color,1)
        cv2.putText(image_to_detect,predicted_class_label,(start_x_point,start_y_point - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color)

    cv2.imshow("Detection Output",image_to_detect)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

file_video_stream.release()
cv2.destroyAllWindows()