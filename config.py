# If true, then no windows will be drawn to the screen - all progress will be shared
# through the terminal window only.
# NOTE: This setting is meant for rendering videos in the background where progress
#       being displayed in real-time is not necessary. This setting is especially
#       useful on terminal-only environments where OpenCV may not be able to open
#       any windows.
TERMINAL_ONLY = False

# Video file to read
VIDEO_PATH = "C:\\Users\\ewelborn\\OneDrive - tarleton.edu (NTNET)\\Research\\Misc Drone Data\\10-10\\west10m.mp4"
#VIDEO_PATH = "thermalSequence4.mp4"

# Do you want the results stored to your computer as a video file? 
# Is overwriting a previous video file okay?
SAVE_RESULT_AS_VIDEO = True
OVERWRITE_PREVIOUS_RESULT = True
SAVE_HOMOGRAPHY_AS_VIDEO = True

# Should the input video be resized? If so, what's the target resolution (width, height)?
# NOTE: This will also affect the resolution of the output video
RESIZE_INPUT_VIDEO = True
RESIZE_INPUT_VIDEO_TARGET_RESOLUTION = (1920, 1080)

# How many processes/threads are allowed to run at a time?
MULTITHREADING_PROCESSES = 2

# Should the GPU be used for processing?
GPU_ENABLED = True

# Video file to write to (no effect if SAVE_RESULT_AS_VIDEO = False)
OUTPUT_VIDEO_PATH = "videos/west10m.mp4"
OUTPUT_HOMOGRAPHY_VIDEO_PATH = "videos/west10m_hom.mp4"

# How many frames should the program process? Set this to -1 to keep processing until
# the end of the video
MAX_FRAMES_TO_PROCESS = 35

# Should the speeds be converted to miles-per-hour? If false, then the speeds will
# be kept in kilometers-per-hour
CONVERT_TO_MPH = False

# What detection model are we using? (YOLO, MASKRCNN)
DETECTION_MODEL = "MASKRCNN"

# What labels are we allowed to track from the COCO dataset?
# This is to ensure we're only tracking vehicles, not other random objects in the frame.
COCO_LABELS_TO_DETECT = ["car", "truck"]

# How confident should our detection model be when detecting vehicles? (0 is no confidence, 1 is completely confident)
DETECTION_MINIMUM_CONFIDENCE = 0.01

# How many frames should we wait until the detection model is allowed to detect cars again?
DETECTION_REFRESH_RATE = 10#15

# Non-maximum suppression is for reducing the likelihood of multiple detections for the same vehicle.
# How much overlap is allowed between two competing detections? (0 is none, 1 is complete overlap)
NMS_THRESHOLD = 0.7

# Where are the YOLOv4 configuration and weight files contained? (No effect if DETECTION_MODEL != "YOLO")
YOLO_CONFIG_PATH = "yolov7.conv.132"
YOLO_WEIGHTS_PATH = "yolov7.weights"

# Where are the Mask-RCNN configuration and weight files contained? (No effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_CONFIG_PATH = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
MASKRCNN_WEIGHTS_PATH = "frozen_inference_graph.pb"

# What threshold value should we use when performing pixelwise segmentation? (Between 0 and 1, no effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_PIXEL_SEGMENTATION_THRESHOLD = 0.2

# Should the detected masks be drawn? If so, what intensity and color should they be drawn with?
# (No effect if DETECTION_MODEL != "maskRCNN")
MASKRCNN_DRAW_MASKS = True
MASKRCNN_DRAW_MASKS_INTENSITY = 0.5
# Color is represented in BGR format
MASKRCNN_DRAW_MASKS_COLOR = (255,0,0)

# What model should we use to track bounding boxes in our model? (Case insensitive)
# Options: BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT (Recommended), GOTURN (Recommended)
# NOTE: GOTURN will use a seperate tracking system - all other options use CV2's legacy MultiTracker system
# NOTE: GOTURN *only* works with GPU enabled
BOUNDING_BOX_TRACKING_MODEL = "CSRT"

# How much overlap is required between the bounding box of a car detected last round versus a car
# detected this round to prove that these two detections are the same car?
MINIMUM_BB_OVERLAP = 0.35

# How fast is the video in frames per second?
FPS = 30

# Should additional information about the algorithm be printed to the console?
DEBUG = False

# Should the time code be printed along with the debug information?
DEBUG_TIMECODE = True

# How should the computer determine the homography for tracking vehicles?
# Options: AUTO, MANUAL, LOADFILE
HOMOGRAPHY_INFERENCE = "manual"

# If HOMOGRAPHY_INFERENCE is MANUAL and HOMOGRAPHY_SAVE_TO_FILE is true, then this is
# the file we save the homography to.
# If HOMOGRAPHY_INFERENCE is LOADFILE, then this is the file we load the homography from.
HOMOGRAPHY_FILE = "homography/west10m.npz"
HOMOGRAPHY_SAVE_TO_FILE = True

# Can we overwrite a pre-existing homography?
HOMOGRAPHY_SAVE_TO_FILE_OVERWRITE = True

# Should we draw what the warped image looks like? Has no effect if HOMOGRAPHY_INFERENCE is AUTO
DRAW_MANUAL_HOMOGRAPHY = True

# Should the bounding boxes and labels for the detected cars be drawn?
DRAW_2D_BOUNDING_BOXES = True
DRAW_3D_BOUNDING_BOXES = False
DRAW_LABELS = True
DRAW_ANGLE_OF_MOTION = False
DRAW_FRAME_COUNT = True

# How many times per second should the label information for each vehicle be updated?
DRAW_LABEL_UPDATE_FREQUENCY = 1

# How thick should the detection rectangles be? (In pixels)
DRAWING_THICKNESS = 3

# How many frames of information should we use to compute the vehicle's speed?
# Less frames = faster response time to sudden change in vehicle speed
# More frames = smoother, less jittery speed estimation of vehicle
# Set this to 1 to disable smoothing
VEHICLE_SPEED_ESTIMATION_SMOOTHING_FRAMES = 4

# How many frames of information should we use to compute the vehicle's 3D bounding box?
# Less frames = faster response time to sudden change in vehicle direction
# More frames = smoother, less jittery estimation of bounding box
# Set this to 1 to disable smoothing
VEHICLE_3D_BOUNDING_BOX_ESTIMATION_SMOOTHING_FRAMES = 5

# By what factor should the homography be scaled? Higher values will be more intensive
# on the PC, lower values result in lower tracking accuracy.
# Recommended: 10
# NOTE: This value is ignored if HOMOGRAPHY_INFERENCE = "loadfile" - the original
# scaling factor from the saved homography file will be used instead
HOMOGRAPHY_SCALING_FACTOR = 16

# At what speed threshold (in KMH or MPH based on the value of CONVERT_TO_MPH) should
# the vehicles be under for the algorithm to say that they're parked?
PARKED_VEHICLE_SPEED_THRESHOLD = 15

# For how many seconds does the vehicle need to stay under this threshold in order
# for the algorithm to flag it as parked?
PARKED_VEHICLE_TIME_THRESHOLD = 2

# Should the window be resizable? If it is, then the results shown on screen
# will not be pixel-accurate - but, the results can be scaled better to your display
RESIZABLE_WINDOW = False