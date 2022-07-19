# Video file to read
VIDEO_PATH = "\\Datasets\\cam_10.mp4"
#VIDEO_PATH = "thermalSequence4.mp4"

# Do you want the results stored to your computer as a video file? 
# Is overwriting a previous video file okay?
SAVE_RESULT_AS_VIDEO = False
OVERWRITE_PREVIOUS_RESULT = False

# Video file to write to (no effect if SAVE_RESULT_AS_VIDEO = False)
OUTPUT_VIDEO_PATH = "boundingBoxes3D_2.mp4"

# What detection model are we using? (YOLO, MASKRCNN)
DETECTION_MODEL = "MASKRCNN"

# What labels are we allowed to track from the COCO dataset?
# This is to ensure we're only tracking vehicles, not other random objects in the frame.
COCO_LABELS_TO_DETECT = ["car", "truck"]

# How confident should our detection model be when detecting vehicles? (0 is no confidence, 1 is completely confident)
DETECTION_MINIMUM_CONFIDENCE = 0.3

# How many frames should we wait until the detection model is allowed to detect cars again?
DETECTION_REFRESH_RATE = 10

# Non-maximum suppression is for reducing the likelihood of multiple detections for the same vehicle.
# How much overlap is allowed between two competing detections? (0 is none, 1 is complete overlap)
NMS_THRESHOLD = 0.7

# Where are the YOLOv4 configuration and weight files contained? (No effect if DETECTION_MODEL != "YOLO")
YOLO_CONFIG_PATH = "yolov4.cfg"
YOLO_WEIGHTS_PATH = "yolov4.weights"

# Where are the Mask-RCNN configuration and weight files contained? (No effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_CONFIG_PATH = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
MASKRCNN_WEIGHTS_PATH = "frozen_inference_graph.pb"

# What threshold value should we use when performing pixelwise segmentation? (Between 0 and 1, no effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_PIXEL_SEGMENTATION_THRESHOLD = 0.2

# Should the detected masks be drawn? If so, what intensity and color should they be drawn with?
# (No effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_DRAW_MASKS = False
MASKRCNN_DRAW_MASKS_INTENSITY = 0.5
# Color is represented in BGR format
MASKRCNN_DRAW_MASKS_COLOR = (255,0,0)

# What model should we use to track bounding boxes in our model? (Case insensitive)
# Options: BOOSTING, MIL, KCF (Recommended), TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT
BOUNDING_BOX_TRACKING_MODEL = "KCF"

# How much overlap is required between the bounding box of a car detected last round versus a car
# detected this round to prove that these two detections are the same car?
MINIMUM_BB_OVERLAP = 0.20

# How fast is the video in frames per second?
FPS = 10

# Should additional information about the algorithm be printed to the console?
DEBUG = True

# Should the time code be printed along with the debug information?
DEBUG_TIMECODE = True

# How should the computer determine the homography for tracking vehicles?
# Options: AUTO, MANUAL, LOADFILE
HOMOGRAPHY_INFERENCE = "loadfile"

# If HOMOGRAPHY_INFERENCE is MANUAL and HOMOGRAPHY_SAVE_TO_FILE is true, then this is
# the file we save the homography to.
# If HOMOGRAPHY_INFERENCE is LOADFILE, then this is the file we load the homography from.
HOMOGRAPHY_FILE = "cam_10.npy"
HOMOGRAPHY_SAVE_TO_FILE = True

# Can we overwrite a pre-existing homography?
HOMOGRAPHY_SAVE_TO_FILE_OVERWRITE = False

# Should we draw what the warped image looks like? Has no effect if HOMOGRAPHY_INFERENCE is AUTO
DRAW_MANUAL_HOMOGRAPHY = True

# Should the bounding boxes and labels for the detected cars be drawn?
DRAW_2D_BOUNDING_BOXES = False
DRAW_3D_BOUNDING_BOXES = True
DRAW_LABELS = True

# How thick should the detection rectangles be? (In pixels)
DRAWING_THICKNESS = 3

# How many frames of information should we use to compute the vehicle's speed?
# Less frames = faster response time to sudden change in vehicle speed
# More frames = smoother, less jittery speed estimation of vehicle
# Set this to 1 to disable smoothing
VEHICLE_SPEED_ESTIMATION_SMOOTHING_FRAMES = 3

# By what factor should the homography be scaled? Higher values will be more intensive
# on the PC, lower values result in lower tracking accuracy.
# Recommended: 10
HOMOGRAPHY_SCALING_FACTOR = 10
