def trackVehicles(batch):
    from MultiTrackerWrapper import MultiTrackerWrapper
    from MaskRCNNDetector import MaskRCNNDetector

    from Worker import Worker

    worker = Worker(MaskRCNNDetector(), MultiTrackerWrapper())

    return {"trackedVehicles": worker.findVehiclesInFrames(batch["frames"]), "index": batch["index"]}

# Guard clause necessary to allow for multiprocessing
if __name__ == "__main__":
    import cv2
    import numpy as np
    import math
    import os
    from multiprocessing import Process, Queue, Pool
    from tqdm import tqdm
    from time import time
    from random import sample

    from stitcher import stitchTrackedVehicles

    from ManualHomographyWrapper import ManualHomographyWrapper
    from StandardVisualizer import StandardVisualizer
    from AutomaticHomographyEstimator import AutomaticHomographyEstimator

    ### CONFIGURATION SETTINGS
    import config

    # Ensure we're using the proper OpenCV package
    if hasattr(cv2, "legacy") == False:
        print("cv2.legacy is missing! Please install (or enable) an OpenCV version that has this package.")
        print("The legacy package is required so that the multitracker object can be used.")
        print("Recommended pip package is 'opencv-contrib-python' (4.6.0.66)")
        exit()

    # Make sure the detection model is valid
    config.DETECTION_MODEL = config.DETECTION_MODEL.upper()
    if not(config.DETECTION_MODEL in ["YOLO", "MASKRCNN"]):
        print(config.DETECTION_MODEL, "is not a valid detection model")
        print("Please try \"YOLO\" or \"MASKRCNN\"")
        exit()

    # Make sure that HOMOGRAPHY_INFERENCE is valid
    config.HOMOGRAPHY_INFERENCE = config.HOMOGRAPHY_INFERENCE.upper()
    if not (config.HOMOGRAPHY_INFERENCE in ["MANUAL", "AUTO", "LOADFILE"]):
        print(config.HOMOGRAPHY_INFERENCE, "is not a valid inference setting")
        print("Please try \"MANUAL\", \"AUTO\", or \"LOADFILE\"")
        exit()
    elif config.HOMOGRAPHY_INFERENCE == "MANUAL" and config.TERMINAL_ONLY:
        print("Manual homography inference is not possible in terminal-only mode.")
        print("Please try one of the following options and restart the program:")
        print("\t- Set TERMINAL_ONLY to False")
        print("\t- Set HOMOGRAPHY_INFERENCE to LOADFILE")
        exit()
    elif config.HOMOGRAPHY_INFERENCE == "AUTO":
        #print("Auto homography inference is not supported at this time.")
        #print("Please set HOMOGRAPHY_INFERENCE to MANUAL or LOADFILE")
        pass

    # If we're loading the homography from a file, ensure that it's loading correctly
    manualHomographyMatrix = None
    if config.HOMOGRAPHY_INFERENCE == "LOADFILE":
        if not os.path.isfile(config.HOMOGRAPHY_FILE):
            print("Could not find homography file:",config.HOMOGRAPHY_FILE)
            exit()

        homographyFile = np.load(config.HOMOGRAPHY_FILE)
        manualHomographyMatrix = homographyFile["manualHomographyMatrix"]
        config.HOMOGRAPHY_SCALING_FACTOR = homographyFile["homographyScalingFactor"]

    # If we're saving the homography to a file, make sure it doesn't already exist!
    #   (Or if it does, make sure we're allowed to overwrite it)
    if config.HOMOGRAPHY_INFERENCE == "MANUAL" and config.HOMOGRAPHY_SAVE_TO_FILE:
        if config.HOMOGRAPHY_SAVE_TO_FILE_OVERWRITE == False and os.path.isfile(config.HOMOGRAPHY_FILE):
            print("Manual homography already exists at path:",config.HOMOGRAPHY_FILE)
            exit()

    # Make sure the tracking model is valid
    trackingModels = {
        "BOOSTING": lambda: cv2.legacy.TrackerBoosting_create(),
        "MIL": lambda: cv2.legacy.TrackerMIL_create(),
        "KCF": lambda: cv2.legacy.TrackerKCF_create(),
        "TLD": lambda: cv2.legacy.TrackerTLD_create(),
        "MEDIANFLOW": lambda: cv2.legacy.TrackerMedianFlow_create(),
        "GOTURN": lambda: cv2.legacy.TrackerGOTURN_create(),
        "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create(),
        "CSRT": lambda: cv2.legacy.TrackerCSRT_create(),
    }
    config.BOUNDING_BOX_TRACKING_MODEL = config.BOUNDING_BOX_TRACKING_MODEL.upper()
    usingLegacyMultiTracker = True
    if config.BOUNDING_BOX_TRACKING_MODEL == "GOTURN":
        # May use for providing information to user
        usingLegacyMultiTracker = False
    elif not (config.BOUNDING_BOX_TRACKING_MODEL in trackingModels.keys()):
        print(config.BOUNDING_BOX_TRACKING_MODEL,"is not a valid tracking model")
        print("Please try any of the following:", " ".join(trackingModels.keys()))
        exit()

    # Make sure the video exists
    if not os.path.isfile(config.VIDEO_PATH):
        print("File at videoPath does not exist:", str(config.VIDEO_PATH))
        exit()

    # Make sure the output video does *not* exist (if applicable)
    if config.SAVE_RESULT_AS_VIDEO and config.OVERWRITE_PREVIOUS_RESULT == False:
        if os.path.isfile(config.OUTPUT_VIDEO_PATH):
            print("Output video already exists at the following path:", str(config.OUTPUT_VIDEO_PATH))
            exit()

    # Make sure the window is resizable
    if config.RESIZABLE_WINDOW:
        cv2.namedWindow("Vehicle Tracking", cv2.WINDOW_NORMAL)

    visualizer = StandardVisualizer()

    # Set up the video capture
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    totalFramesInVideo = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # back-left, back-right, front-right, front-left
    # https://www.thezebra.com/resources/driving/average-car-size/
    # Average car length: 14.7 feet --> 4.48 meters
    # Average car width: 5.8 feet --> 1.77 meters
    MEAN_SEDAN_LENGTH = 4.48
    MEAN_SEDAN_WIDTH = 1.77
    MEAN_SEDAN_BOUNDING_BOX = [
        [0, 0],
        [MEAN_SEDAN_WIDTH * config.HOMOGRAPHY_SCALING_FACTOR, 0],
        [MEAN_SEDAN_WIDTH * config.HOMOGRAPHY_SCALING_FACTOR, MEAN_SEDAN_LENGTH * config.HOMOGRAPHY_SCALING_FACTOR],
        [0, MEAN_SEDAN_LENGTH * config.HOMOGRAPHY_SCALING_FACTOR]
    ]

    def debugPrint(*args):
        if config.DEBUG:
            if config.DEBUG_TIMECODE:
                global totalFrames
                timeSignature = "({seconds}:{frames})".format(seconds=totalFrames//config.FPS, frames=totalFrames%config.FPS)
                print(timeSignature,*args)
            else:
                print(*args)

    currentCars = []
    oldCars = []
    currentCarID = 0

    # Takes in point p in the form of (x, y) and matrix is a 3x3 homography matrix
    # https://stackoverflow.com/questions/57399915/how-do-i-determine-the-locations-of-the-points-after-perspective-transform-in-t
    def findPointInWarpedImage(p, matrix):
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        return (int(px), int(py))

    # Only used with manual homography
    manualHomography_pixelsToMeters = 0
    manualHomography_pointSize = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Read the first frame of the video - we may need it for homography selection
    ret, initialFrame = cap.read()

    if config.RESIZE_INPUT_VIDEO:
        initialFrame = cv2.resize(initialFrame, config.RESIZE_INPUT_VIDEO_TARGET_RESOLUTION)

    frameWidth = initialFrame.shape[1]
    frameHeight = initialFrame.shape[0]

    if config.HOMOGRAPHY_INFERENCE == "MANUAL":
        if config.HOMOGRAPHY_SAVE_TO_FILE:
            print("The following selection will be saved to disk at", config.HOMOGRAPHY_FILE, "\n")

        print("Please select the four corners of a car in the frame.")
        print("BLUE: Front left headlight")
        print("RED: Front right headlight")
        print("GREEN: Back left tail-light")
        print("ORANGE: Back right tail-light")
        print("\nWhen you are done, press SPACE to continue.")

        manualHomography_points = [[50,50],[100,50],[100,100],[50,100]]
        manualHomography_selectedPoint = None
        def onMouse(event, x, y, flags, params):
            global manualHomography_selectedPoint
            selectedDistance = 0

            newImage = initialFrame.copy()

            if event == cv2.EVENT_LBUTTONDOWN:
                manualHomography_selectedPoint = None
                for point in manualHomography_points:
                    distance = math.hypot(x - point[0], y - point[1])
                    if math.hypot(x - point[0], y - point[1]) <= manualHomography_pointSize:
                        if manualHomography_selectedPoint and selectedDistance <= distance:
                            pass
                        else:
                            manualHomography_selectedPoint = point
                            selectedDistance = distance
            elif event == cv2.EVENT_LBUTTONUP:
                manualHomography_selectedPoint = None
                selectedDistance = 0

            if manualHomography_selectedPoint != None:
                manualHomography_selectedPoint[0] = x
                manualHomography_selectedPoint[1] = y

            for i in range(len(manualHomography_points)):
                point = manualHomography_points[i]
                color = [(255,100,100),(100,100,255),(100,175,255),(100,255,100)][i]
                cv2.circle(newImage,(point[0], point[1]),manualHomography_pointSize,color,-1)
                cv2.circle(newImage,(point[0], point[1]),manualHomography_pointSize-4,[c/3 for c in color],-1)

            cv2.imshow("Vehicle Tracking", newImage)

        cv2.imshow("Vehicle Tracking", initialFrame)
        cv2.setMouseCallback("Vehicle Tracking", onMouse)
        inputKey = None
        while inputKey != ord(" "):
            inputKey = cv2.waitKey(0) & 0xFF

        # Reset the mouse callback so that it doesn't do anything
        cv2.setMouseCallback("Vehicle Tracking", lambda event, x, y, flags, params: 0)

        # https://www.thezebra.com/resources/driving/average-car-size/
        # Average car length: 14.7 feet
        # Average car width: 5.8 feet
        carReferencePoints = np.array(MEAN_SEDAN_BOUNDING_BOX) + np.array([[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2],[frameWidth//2, frameHeight//2]])
        manualHomographyMatrix, status = cv2.findHomography(np.array(manualHomography_points), carReferencePoints)

        print("Homography calculated.")

        # Save the homography matrix if applicable
        if config.HOMOGRAPHY_SAVE_TO_FILE:
            np.savez(config.HOMOGRAPHY_FILE, manualHomographyMatrix=manualHomographyMatrix, homographyScalingFactor=config.HOMOGRAPHY_SCALING_FACTOR)
            print("Homography saved.")

    print(f"Loading frames into memory (from video file {config.VIDEO_PATH}) and distributing amongst subprocesses...")

    # Read all of the frames into memory
    #   (this is either genius, or really dumb, I don't know which)
    frames = []
    totalFrames = 0
    while True:
        ret, frame = cap.read()

        # Keep running until we've processed all of the frames that the user requested,
        #   or we reached the end of the video.
        videoEnd = (ret == False) or ((totalFrames >= config.MAX_FRAMES_TO_PROCESS) and config.MAX_FRAMES_TO_PROCESS > -1)

        if videoEnd == False:
            totalFrames += 1
            if config.RESIZE_INPUT_VIDEO:
                frame = cv2.resize(frame, config.RESIZE_INPUT_VIDEO_TARGET_RESOLUTION)

            frames.append(frame)
        else:
            cap.release()
            break

    # Separate the frames into batches and distribute them to our pool
    pool = Pool(processes=config.MULTITHREADING_PROCESSES)

    # Give n frames to each batch, with 1 frame of overlap between each batch.
    n = config.DETECTION_REFRESH_RATE
    batches = [(i * (n - 1), (i + 1) * (n - 1)) for i in range(len(frames) // (n - 1))]
    batches.append( ( (len(frames) // (n - 1)) * (n - 1), len(frames) - 1 ) )

    # Convert the indices to slices
    batches = [frames[a:b+1] for (a, b) in batches]

    # Add tags for sorting after processing
    for i in range(len(batches)):
        batches[i] = {"frames": batches[i], "index":i}

    print(f"Frames loaded, detecting and tracking vehicles in parallel ({config.MULTITHREADING_PROCESSES} processes)... ")

    # Use TQDM to make a nice progress bar for the parallel tracking
    results = []
    for result in tqdm(pool.imap_unordered(trackVehicles, batches), total=len(batches), colour="blue", unit=f"batch ({n} frames)"):
        results.append(result)

    results.sort(key=lambda b: b["index"])

    pool.close()

    print("Subprocess processing complete, stitching results together...")

    trackedVehicles = stitchTrackedVehicles([b["trackedVehicles"] for b in results])

    for vehicle in trackedVehicles:
        if vehicle.lastFrame is None:
            vehicle.lastFrame = totalFrames

    print("Stitching complete, estimating speed (and other things too!)")

    # Now, we can estimate homography and estimate speed
    homographyEstimator = None
    pixelToMeterRatio = config.HOMOGRAPHY_SCALING_FACTOR
    if config.HOMOGRAPHY_INFERENCE in ["MANUAL", "LOADFILE"]:
        homographyEstimator = ManualHomographyWrapper(manualHomographyMatrix)
    else:
        homographyEstimator = AutomaticHomographyEstimator()

    # Select 5 frames to estimate homography on randomly from the video
    randomFrames = sample(frames, k=5)
    #homography = homographyEstimator.getHomographyFromFrames(randomFrames)
    homography = homographyEstimator.getHomographyFromFrame(initialFrame)

    if config.HOMOGRAPHY_INFERENCE == "AUTO":
        # Estimate the pixel to meter ratio
        samples = []

        for vehicle in trackedVehicles:
            maskHistory = vehicle.getMaskHistory()
            bbHistory = vehicle.getBoundingBoxHistory()
            for i in range(len(maskHistory)):
                mask = maskHistory[i]

                # Only perform estimation on the original masks produced by Mask R-CNN
                if mask.isDuplicate():
                    continue

                # Estimate the angle of motion using past and future frames (if they exist)
                # NOTE: Replace magic number with a config value
                pastBB = bbHistory[max(i - 6, 0)]
                futureBB = bbHistory[min(i + 6, len(bbHistory) - 1)]
                dx = pastBB.getCenterX() - futureBB.getCenterX()
                dy = pastBB.getEndY() - futureBB.getEndY()

                totalMovement = ((dx ** 2) + (dy ** 2)) ** 0.5

                angleOfMotion = math.atan2(dy, dx)

                # Only perform estimation on moving vehicles... If the vehicle is standing still,
                #   then ignore it. The angle of motion will be erroneous.
                if totalMovement <= 15:
                    continue

                # If the vehicle is moving horizontally or vertically in the frame, then we can simply
                #   take the bottom edge of the bounding box and use it to determine the pixel-to-meter
                #   ratio. If the vehicle is moving diagonally, then we'll try to find the tires

                # WARNING: magic number, move to config later
                thresholdInRadians = math.radians(0)

                if (
                    math.isclose(angleOfMotion, 0, abs_tol=thresholdInRadians) or 
                    math.isclose(angleOfMotion, math.pi, abs_tol=thresholdInRadians) or 
                    math.isclose(angleOfMotion, -math.pi, abs_tol=thresholdInRadians)
                ):
                    # Going horizontally
                
                    bb = bbHistory[i]
                    point1 = findPointInWarpedImage((bb.getX(), bb.getY()), homography)
                    point2 = findPointInWarpedImage((bb.getEndX(), bb.getY()), homography)
                    vehicleLength = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
                    sample = vehicleLength / MEAN_SEDAN_LENGTH

                    #print("pixelToMeterRatio sample (horizontally):", sample)
                    #samples.append(sample)

                elif (
                    math.isclose(angleOfMotion, math.pi / 2, abs_tol=thresholdInRadians) or 
                    math.isclose(angleOfMotion, -math.pi / 2, abs_tol=thresholdInRadians)
                ):
                    # Going vertically
                
                    bb = bbHistory[i]
                    point1 = findPointInWarpedImage((bb.getX(), bb.getEndY()), homography)
                    point2 = findPointInWarpedImage((bb.getEndX(), bb.getEndY()), homography)
                    vehicleWidth = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
                    sample = vehicleWidth / MEAN_SEDAN_WIDTH

                    #print("pixelToMeterRatio sample (vertically):", sample)
                    #samples.append(sample)

                else:
                    # Going diagonally-ish

                    indexArray = np.indices(mask.getOriginalMask().shape)

                    # indexArray's shape is currently (2, width, height), which does not work for
                    #   the calculations we need to do. We're transposing it to (width, height, 2)
                    indexArray = np.transpose(indexArray, (1, 2, 0))

                    v = np.array([math.sin(angleOfMotion), math.cos(angleOfMotion)])

                    distanceArray = indexArray @ v

                    mn = np.min(distanceArray)
                    r = np.max(distanceArray) - mn

                    distanceArray = np.where(mask.getOriginalMask(), distanceArray, -np.inf)
                        
                    # Get only the points that are within 5% of the distance of the farthest point (if that makes sense)
                    farthestPoints = np.argwhere(distanceArray >= np.max(distanceArray) - (r * 0.05))

                    # Add the distance as the third column
                    distanceVector = [distanceArray[p[0]][p[1]] for p in farthestPoints]
                    farthestPoints = np.append(farthestPoints, np.array(distanceVector).reshape((len(distanceVector), 1)), axis=1)
            
                    # Sort by distance
                    ind = np.argsort(farthestPoints[:,-1])
                    farthestPoints = farthestPoints[ind]

                    farthestPoint = farthestPoints[0]

                    tire1 = farthestPoint
                    tire2 = None

                    # Try and find the other tire. It should be the farthest point that is *not* within a certain
                    #   range of the first tire
                    for point in farthestPoints:
                        if ((point[0] - farthestPoint[0]) ** 2 + (point[1] - farthestPoint[1]) ** 2) ** 0.5 > 15:
                            tire2 = point
                            break

                    if not (tire2 is None):
                        # X, Y coordinates for points
                        tire1Warped = findPointInWarpedImage((tire1[1], tire1[0]), homography)
                        tire2Warped = findPointInWarpedImage((tire2[1], tire2[0]), homography)

                        sample = ((tire1Warped[0] - tire2Warped[0]) ** 2 + (tire1Warped[1] - tire2Warped[1]) ** 2) ** 0.5
                        sample /= MEAN_SEDAN_LENGTH

                        print("pixelToMeterRatio sample:", sample)
                        samples.append(sample)

        pixelToMeterRatio = np.median(np.array([samples]))
        print("Predicted pixel to meter ratio V.S. given config ratio:", pixelToMeterRatio, config.HOMOGRAPHY_SCALING_FACTOR)
        print("Total samples:", len(samples))

    # Estimate the vehicle's speed based on the movement of the bounding box in the top-down image
    for vehicle in trackedVehicles:
        bbHistory = vehicle.getBoundingBoxHistory()
        for i in range(len(bbHistory)):
            # Select the bounding boxes that we want to compare
            #oldBB = bbHistory[max(i - (config.VEHICLE_SPEED_ESTIMATION_SMOOTHING_FRAMES) // 2, 0)]
            #newBB = bbHistory[min(i + (config.VEHICLE_SPEED_ESTIMATION_SMOOTHING_FRAMES+1) // 2, len(bbHistory) - 1)]
            oldBB = bbHistory[max(i - 1, 0)]
            newBB = bbHistory[min(i, len(bbHistory) - 1)]

            oldX, oldY = findPointInWarpedImage((oldBB.getCenterX(), oldBB.getY() + oldBB.getHeight()), homography)
            newX, newY = findPointInWarpedImage((newBB.getCenterX(), newBB.getY() + newBB.getHeight()), homography)
            dx = newX - oldX
            dy = newY - oldY

            totalMovementInPixels = math.sqrt((dx * dx) + (dy * dy))

            metersPerSecond = (totalMovementInPixels * config.FPS) / pixelToMeterRatio
            #car.KMH = (metersPerSecond * 60 * 60) / 1000
            vehicle.recordKMH((metersPerSecond * 60 * 60) / 1000)

        # Smooth the speed
        vehicle.KMHHistory = np.convolve(
            vehicle.KMHHistory,
            np.array( [1/config.VEHICLE_SPEED_ESTIMATION_SMOOTHING_FRAMES for i in range(config.VEHICLE_SPEED_ESTIMATION_SMOOTHING_FRAMES)] )
        ).tolist()

    # Draw the results to video
    print("Speed estimation complete, drawing results to output")
    newFrames = visualizer.drawFrames(frames, trackedVehicles)

    if config.SAVE_RESULT_AS_VIDEO:
        out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, config.FPS, (frameWidth, frameHeight))
        for frame in newFrames:
            out.write(frame)

        out.release()

        print(f"Video saved to {config.OUTPUT_VIDEO_PATH}")

    if config.SAVE_HOMOGRAPHY_AS_VIDEO:
        # https://stackoverflow.com/a/59741739
        corners = []
        corners.append(findPointInWarpedImage((0, 0), homography))
        corners.append(findPointInWarpedImage((frameWidth, 0), homography))
        corners.append(findPointInWarpedImage((0, frameHeight), homography))
        corners.append(findPointInWarpedImage((frameWidth, frameHeight), homography))

        print("corners",corners)

        xmin = min(p[0] for p in corners)
        ymin = min(p[1] for p in corners)
        xmax = max(p[0] for p in corners)
        ymax = max(p[1] for p in corners)

        width = max(p[0] for p in corners)# - min(p[0] for p in corners)
        height = max(p[1] for p in corners)# - min(p[1] for p in corners)
        scale = min(frameWidth / width, frameHeight / height)

        print("width",width)
        print("height",height)
        print("scale",scale)

        out = None

        for frame in frames:
            warpedFrame = cv2.warpPerspective(frame, homography, (width, height))

            resizedFrame = cv2.resize(warpedFrame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            if out == None:
                out = cv2.VideoWriter(config.OUTPUT_HOMOGRAPHY_VIDEO_PATH, fourcc, config.FPS, (resizedFrame.shape[1], resizedFrame.shape[0]))

            out.write(resizedFrame)

        out.release()
        print(f"Homography video saved to {config.OUTPUT_HOMOGRAPHY_VIDEO_PATH}")

    print("Processing complete")