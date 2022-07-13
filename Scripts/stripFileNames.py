# Quick and dirty script for stripping off timecodes from the ends of file names
# Ex. frame_1_0_0.57833.jpg => frame_1_0.jpg
import os
for root, dirs, files in os.walk("thermal_13Aug2020"):
    for file in files:
        components = file.split("_")
        newFile = components[0] + "_" + components[1] + "_" + components[2] + ".jpg"
        os.rename("thermal_13Aug2020//" + file, "thermal_13Aug2020//" + newFile)
