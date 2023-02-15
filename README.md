# PERS-grant-tracking-vehicles
*Still in search of a good name 😊*

This program estimates the speed of vehicles in a video. It achieves this by detecting vehicles with a neural network (YOLOv4 or Mask R-CNN), tracking the vehicles (OpenCV's MultiTracker class), and estimating the distance travelled using a homography transformation.

## Additional Files
This repository does not contain all of the files necessary to run the project, due to licensing and upload limitations. These files can be downloaded separately by following the links below.

### Mask R-CNN
To use the Mask R-CNN detection algorithm, you'll need the config file (a description of the neural network) as well as a weights file. [Both files can be found here](https://github.com/sambhav37/Mask-R-CNN/tree/master/mask-rcnn-coco) - the network is pretrained on the [MS COCO dataset](https://cocodataset.org/#home), and should work for detecting vehicles.
To enable Mask R-CNN, make the following updates to your ```config.py``` file.
```python
DETECTION_MODEL = "MASKRCNN" # <-- Update to "MASKRCNN"
# ...
MASKRCNN_CONFIG_PATH = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt" # <-- Update so that the path points to your config file
MASKRCNN_WEIGHTS_PATH = "frozen_inference_graph.pb" # <-- Update so that the path points to your weights file
```

### Bird's Eye View
To predict a top-down view of the road, our algorithm uses the deep neural network provided by Abbas and Zisserman's work. You will need to [download their GitHub repository here](https://github.com/SAmmarAbbas/birds-eye-view), and store it as a folder inside of our repository. Additionally, [you may find their paper here](https://arxiv.org/abs/1905.02231).
