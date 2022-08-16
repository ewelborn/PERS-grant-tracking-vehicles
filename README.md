# PERS-grant-tracking-vehicles
*Still in search of a good name 😊*

This program estimates the speed of vehicles in a video. It achieves this by detecting vehicles with a neural network (YOLOv4 or Mask R-CNN), tracking the vehicles (OpenCV's MultiTracker class), and estimating the distance travelled using a homography transformation.

## Additional Files
This repository does not contain all of the files necessary to run the project, due to licensing and upload limitations. These files can be downloaded separately by following the links below.

### YOLOv4
To use the YOLOv4 detection algorithm, you'll need the config file (a description of the neural network) as well as a weights file. [Both files can be found here](https://github.com/AlexeyAB/darknet) - the network is pretrained on the [MS COCO dataset](https://cocodataset.org/#home), and should work for detecting vehicles.
To enable YOLOv4, make the following updates to your ```config.py``` file.
```python
DETECTION_MODEL = "YOLO" # <-- Update to "YOLO"
# ...
YOLO_CONFIG_PATH = "yolov4.cfg" # <-- Update so that the path points to your config file
YOLO_WEIGHTS_PATH = "yolov4.weights" # <-- Update so that the path points to your weights file
```

### Mask R-CNN
To use the Mask R-CNN detection algorithm, you'll need the config file (a description of the neural network) as well as a weights file. [Both files can be found here](https://github.com/sambhav37/Mask-R-CNN/tree/master/mask-rcnn-coco) - the network is pretrained on the [MS COCO dataset](https://cocodataset.org/#home), and should work for detecting vehicles.
To enable Mask R-CNN, make the following updates to your ```config.py``` file.
```python
DETECTION_MODEL = "MASKRCNN" # <-- Update to "MASKRCNN"
# ...
MASKRCNN_CONFIG_PATH = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt" # <-- Update so that the path points to your config file
MASKRCNN_WEIGHTS_PATH = "frozen_inference_graph.pb" # <-- Update so that the path points to your weights file
```
